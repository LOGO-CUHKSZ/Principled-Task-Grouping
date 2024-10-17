import os.path
from logging import getLogger

import math
from collections import defaultdict

import torch
import random
from Env.COPEnv import COPEnv as Env
from Models.models import COPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

# import ray
from utils import *
from weight_methods import WeightMethods
import pickle
import torch.distributed as dist
from copy import deepcopy
from itertools import combinations


def get_inner_model(model):
    return model.module if isinstance(model, DDP) else model


def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
            ),
            stl=dict(main_task=args.main_task),
            cagrad=dict(c=args.c),
            dwa=dict(temp=args.dwa_temp),
        )
    )
    return weight_methods_parameters


def compute_l(x, q, rho, n_tasks):
    kk = 1 / (x + 1)
    q_kk = [math.pow(i, kk) for i in q]
    t1 = sum(q_kk)
    t2 = sum([math.log(q[i]) * q_kk[i] for i in range(len(q))]) / (x + 1)
    return math.log(n_tasks) - rho - math.log(t1) + t2 / t1


# Algorithm 2 in paper
def find_lambda(e, beta, upper, jump, q_k, rho, n_tasks):
    if compute_l(0, q_k, rho, n_tasks) <= 0:
        return 0
    left = 0
    right = beta
    flag = 0
    while compute_l(right, q_k, rho, n_tasks) > 0:
        flag += 1
        left = right
        right = right + beta
        if right > upper:
            return upper
        if flag > jump:
            break
    x = (left + right) / 2
    ans = compute_l(x, q_k, rho, n_tasks)
    flag = 0
    while abs(ans) > e:
        flag += 1
        if ans > 0:
            left = x
        else:
            right = x
        x = (left + right) / 2
        ans = compute_l(x, q_k, rho, n_tasks)
        if flag > jump:  # if lambda is too large, skip out the loop
            return upper
    return x


class Trainer:
    def __init__(
        self, env_params, model_params, optimizer_params, trainer_params, rank, opts
    ):
        self.hfai_mode = opts.hfai_mode
        self.alg = opts.alg
        self.ddp = opts.ddp
        self.tag = opts.tag

        # save arguments
        self.env_params = env_params["seen"]
        self.unseen_params = env_params["unseen"]

        self.evaluation_size = opts.evaluation_size

        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.rank = rank

        self.total_count = 0

        # result folder, logger
        self.logger = getLogger(name="trainer")
        if self.rank != 0:
            self.logger.disabled = True
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        if self.rank == 0 and not os.path.exists(
            os.path.join(self.result_folder, "args.json")
        ):
            with open(os.path.join(self.result_folder, "args.json"), "w") as f:
                json.dump(vars(opts), f, indent=True)

        # cuda
        USE_CUDA = self.trainer_params["use_cuda"]
        if USE_CUDA:
            cuda_device_num = rank
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")

        # Main Components
        self.problem = list(self.env_params.keys())
        self.problem.remove("same")
        self.unseen_problem = list(self.unseen_params.keys())

        self.model = Model(self.problem, **self.model_params)
        self.backup_model = Model(self.problem, **self.model_params)

        self.env_list = Env(**self.env_params).env_list
        self.unseen_env_list = Env(**self.unseen_params).env_list

        # historical best params
        self.hist_best_model_params_seen = [
            [[self.total_count, deepcopy(self.model.state_dict())] for env in cop_env]
            for cop_env in self.env_list
        ]

        self.hist_best_model_params_unseen = [
            [[self.total_count, deepcopy(self.model.state_dict())] for env in cop_env]
            for cop_env in self.unseen_env_list
        ]
        num_tasks = sum([len(cop_env) for cop_env in self.env_list])
        weight_method_params = extract_weight_method_parameters_from_args(opts)

        if opts.alg == "naive":
            self.weighted_method = None
            self.optimizer = Optimizer(
                self.model.parameters(), **self.optimizer_params["optimizer"]
            )
            # self.backup_optimizer = Optimizer(
            #     self.model.parameters(), **self.optimizer_params["optimizer"]
            # )
        elif opts.alg == "banditmtl":
            self.weighted_method = [1 / num_tasks for i in range(num_tasks)]
            self.rho = opts.rho
            self.eta_p = opts.eta_p
            self.optimizer = Optimizer(
                self.model.parameters(), **self.optimizer_params["optimizer"]
            )
            # self.backup_optimizer = Optimizer(
            #     self.model.parameters(), **self.optimizer_params["optimizer"]
            # )
        elif opts.alg == "autol":
            num_tasks = sum([len(cop_env) for cop_env in self.env_list])
            self.meta_weights = torch.tensor([opts.autol_init] * num_tasks, requires_grad=True, device=self.rank)
            self.meta_weight_ls = np.zeros([self.trainer_params["epochs"], num_tasks], dtype=np.float32)
            self.meta_optimizer = torch.optim.Adam([self.meta_weights], lr=opts.autol_lr)
            self.optimizer = Optimizer(
                self.model.parameters(), **self.optimizer_params["optimizer"]
            )
        else:
            self.weighted_method = WeightMethods(
                opts.alg,
                num_tasks,
                device=self.rank,
                **weight_method_params[opts.alg],
            )
            self.optimizer = Optimizer(
                [
                    {
                        **dict(params=self.model.parameters()),
                        **self.optimizer_params["optimizer"],
                    },
                    dict(
                        params=self.weighted_method.parameters(),
                        lr=opts.method_params_lr,
                    ),
                ]
            )

        self.weighted_collection = []
        if opts.alg == "banditmtl":
            self.weighted_collection.append(self.weighted_method)
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params["scheduler"])
        self.start_epoch = 1

        self.eval_res = []
        self.training_time = []
        self.count_each_task = [0 for i in range(num_tasks)]
        self.selected_tasks = []

        self.overall_seen_data = []
        for i, cop_env in enumerate(self.env_list):
            self.overall_seen_data.append([])
            for j, env in enumerate(cop_env):
                generate_data = env.generate_data(
                    opts.evaluation_size
                )
                self.overall_seen_data[-1].append(generate_data)

        self.overall_unseen_data = []
        for i, cop_env in enumerate(self.unseen_env_list):
            self.overall_unseen_data.append([])
            for j, env in enumerate(cop_env):
                generate_data = env.generate_data(
                    opts.evaluation_size
                )
                self.overall_unseen_data[-1].append(generate_data)

        # Restore
        model_load = trainer_params["model_load"]
        if model_load["enable"]:
            self.logger.info("Saved Model Loaded !!")
            try:
                checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(
                    **model_load
                )
                checkpoint = torch.load(checkpoint_fullname, map_location=device)
                load_epoch = model_load["epoch"]
            except:
                epochs = []
                for file in os.listdir(model_load["path"]):
                    if file.split("-")[0] == "checkpoint":
                        try:
                            epochs.append(int(file.split("-")[1].split(".")[0]))
                        except:
                            pass
                load_epoch = max(epochs)
                checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(
                    path=model_load["path"], epoch=load_epoch
                )
                checkpoint = torch.load(checkpoint_fullname, map_location=device)

            self.result_folder = model_load["path"]
            if self.rank == 0:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.result_log.set_raw_data(checkpoint["result_log"])

            self.start_epoch = 1 + checkpoint["epoch"]
            self.total_count = checkpoint["total_count"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.last_epoch = checkpoint["epoch"] - 1
            self.transfer_gain_collection = checkpoint["transfer_gain"]
            self.count_each_task = checkpoint['count_each_task']
            self.selected_tasks = checkpoint['selected_tasks']

            # self.weighted_method = checkpoint['weighted_method']
            # self.weighted_collection = checkpoint['weighted_collection']
            # load resume info for bandit algorithm
            # with open('{}/bandit_info-{}.pkl'.format(model_load['path'],load_epoch), 'rb') as file:
            #     self.bandit = pickle.load(file)

            self.eval_res = checkpoint["eval_res"]
            # self.overall_seen_data = checkpoint["overall_seen_data"]
            # self.overall_unseen_data = checkpoint["overall_unseen_data"]
            # historical best params
            # self.hist_best_model_params_seen = checkpoint["hist_best_model_params_seen"]
            # self.hist_best_model_params_unseen = checkpoint[
            #     "hist_best_model_params_unseen"
            # ]

            self.training_time = checkpoint["training_time"]
            self.collect_time = checkpoint['collect_time']
        # # fix the validation data and send to different gpus
        if rank == 0:
        #     if opts.ddp:
        #         self.fix_seen_validation_data = []
        #         for i, cop_env in enumerate(self.env_list):
        #             self.fix_seen_validation_data.append([])
        #             for j, env in enumerate(cop_env):
        #                 generate_data = self.overall_seen_data[i][j].cuda()
        #                 generate_data_list = torch.chunk(
        #                     generate_data, dist.get_world_size()
        #                 )
        #                 for _ in range(1, dist.get_world_size()):
        #                     dist.send(
        #                         generate_data_list[_], dst=_, tag=i * 100 + j * 10 + _
        #                     )

        #                 self.fix_seen_validation_data[-1].append(generate_data_list[0])

        #     self.fix_unseen_validation_data = []
        #     for i, cop_env in enumerate(self.unseen_env_list):
        #         self.fix_unseen_validation_data.append([])
        #         for j, env in enumerate(cop_env):
        #             generate_data = self.overall_unseen_data[i][j].cuda()
        #             generate_data_list = torch.chunk(
        #                 generate_data, dist.get_world_size()
        #             )
        #             for _ in range(1, dist.get_world_size()):
        #                 dist.send(
        #                     generate_data_list[_],
        #                     dst=_,
        #                     tag=1000 + i * 100 + j * 10 + _,
        #                 )
        #             self.fix_unseen_validation_data[-1].append(generate_data_list[0])

            self.transfer_gain_collection = {tuple([i]): [] for i in range(num_tasks)}
            self.transfer_gain_collection.update(
                {
                    tuple([i, j]): []
                    for i in range(num_tasks)
                    for j in range(num_tasks)
                    if i <= j
                }
            )
            self.transfer_gain_collection["no_lookahead"] = []
            self.collect_time = {tuple([i]): {"backward": [], "forward": [], "update": []} for i in range(num_tasks)}
            self.collect_time.update(
                {
                    tuple([i, j]): {"backward": [], "forward": [], "update": []}
                    for i in range(num_tasks)
                    for j in range(num_tasks)
                    if i <= j
                }
            )

        # else:
        #     self.fix_seen_validation_data = []
        #     for i, cop_env in enumerate(self.env_list):
        #         self.fix_seen_validation_data.append([])
        #         for j, env in enumerate(cop_env):
        #             generate_data = env.generate_data(opts.evaluation_size)
        #             dist.recv(generate_data, src=0, tag=i * 100 + j * 10 + self.rank)
        #             self.fix_seen_validation_data[-1].append(generate_data)

        #     self.fix_unseen_validation_data = []
        #     for i, cop_env in enumerate(self.unseen_env_list):
        #         self.fix_unseen_validation_data.append([])
        #         for j, env in enumerate(cop_env):
        #             generate_data = env.generate_data(opts.evaluation_size)
        #             dist.recv(
        #                 generate_data, src=0, tag=1000 + i * 100 + j * 10 + self.rank
        #             )
        #             self.fix_unseen_validation_data[-1].append(generate_data)

        # if len(self.env_list)!=1 or self.separte_train:
        #     self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        # else:
        #     self.model = DDP(self.model, device_ids=[rank])
        if opts.ddp:
            self.backup_model = DDP(self.backup_model, device_ids=[rank])
            self.model = DDP(self.model, device_ids=[rank])
        # self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        self.lookahead_mode = opts.lookahead_mode
        self.lookahead_freq = opts.lookahead_freq
        self.sampling = opts.sampling
        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            self.logger.info(
                "================================================================="
            )

            # LR Decay
            self.scheduler.step()

            # Train
            total_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append("train_score", epoch, total_score)
            self.result_log.append("train_loss", epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                epoch, self.trainer_params["epochs"]
            )
            self.logger.info(
                "Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                    epoch,
                    self.trainer_params["epochs"],
                    elapsed_time_str,
                    remain_time_str,
                )
            )

            all_done = epoch == self.trainer_params["epochs"]
            model_save_interval = self.trainer_params["logging"]["model_save_interval"]

            # Save Model
            if self.rank == 0 and (all_done or (epoch % model_save_interval) == 0):
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_inner_model(self.model).state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "result_log": self.result_log.get_raw_data(),
                    "eval_res": self.eval_res,
                    # "overall_seen_data": self.overall_seen_data,
                    # "overall_unseen_data": self.overall_unseen_data,
                    # "hist_best_model_params_seen": self.hist_best_model_params_seen,
                    # "hist_best_model_params_unseen": self.hist_best_model_params_unseen,
                    "training_time": self.training_time,
                    "collect_time": self.collect_time,
                    "total_count": self.total_count,
                    "transfer_gain": self.transfer_gain_collection,
                    # 'weighted_method': self.weighted_method,
                    # 'weighted_collection': self.weighted_collection,
                    "count_each_task":self.count_each_task,
                    "selected_tasks":self.selected_tasks
                }
                torch.save(
                    checkpoint_dict,
                    "{}/checkpoint-{}.pt".format(self.result_folder, epoch),
                )
                # with open('{}/bandit_info-{}.pkl'.format(self.result_folder,epoch), 'wb') as file:
                #     pickle.dump(self.bandit, file)


            if self.rank == 0:
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_inner_model(self.model).state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "result_log": self.result_log.get_raw_data(),
                    "eval_res": self.eval_res,
                    # "overall_seen_data": self.overall_seen_data,
                    # "overall_unseen_data": self.overall_unseen_data,
                    # "hist_best_model_params_seen": self.hist_best_model_params_seen,
                    # "hist_best_model_params_unseen": self.hist_best_model_params_unseen,
                    "training_time": self.training_time,
                    "collect_time": self.collect_time,
                    "total_count": self.total_count,
                    "transfer_gain": self.transfer_gain_collection,
                    "count_each_task":self.count_each_task,
                    "selected_tasks":self.selected_tasks
                    # 'weighted_method': self.weighted_method,
                    # 'weighted_collection': self.weighted_collection
                }
                torch.save(
                    checkpoint_dict,
                    "{}/checkpoint-latest.pt".format(self.result_folder),
                )
                with open('{}/time-{}.pkl'.format(self.result_folder,epoch), 'wb') as file:
                    pickle.dump(self.collect_time, file)
                # with open('{}/bandit_info-latest.pkl'.format(self.result_folder), 'wb') as file:
                #     pickle.dump(self.bandit, file)

            # # Save Image
            # if all_done or (epoch % img_save_interval) == 0:
            #     image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
            #                         self.result_log, labels=['train_score'])
            #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
            #                         self.result_log, labels=['train_loss'])

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        # score_AM = AverageMeter()
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params["train_episodes"]
        episode = 0
        loop_cnt = 0
        num_tasks = sum([len(cop_env) for cop_env in self.env_list])
        s = time.time()
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params["train_batch_size"], remaining)
            # if self.rank == 0:
            #     # lookahead phase
            #     if self.sampling == "uniform":
            #         # random sample the task number
            #         k = random.randint(1, num_tasks)
            #         selected_subset = random.sample(list(range(num_tasks)), k,)
            #     elif self.sampling == "bias":
            #         k = random.randint(1, num_tasks)
            #         normalized_count = np.array(self.count_each_task)/(np.sum(self.count_each_task)+1e-8)
            #         p = np.exp(-normalized_count)
            #         p = p / p.sum()
            #         selected_subset = np.random.choice(list(range(num_tasks)), size=k, replace=False, p=p)
            #     else:
            #         selected_subset = list(range(num_tasks))
                    
            #     self.selected_tasks.append(selected_subset)
            #     # distributed the selected subset
            #     masked_selected_subset = [i if i in selected_subset else -1 for i in range(num_tasks)]
            #     masked_selected_subset_tensor = torch.tensor(masked_selected_subset)
            # else:
            #     masked_selected_subset_tensor = torch.empty(num_tasks, dtype=torch.long)
            # if self.ddp:
            #     dist.broadcast(masked_selected_subset_tensor, src=0)
            # # selected subset is not -1
            # selected_subset = [i for i in masked_selected_subset_tensor.tolist() if i != -1]
            # selected_subset.sort()
            # self.count_each_task = [self.count_each_task[i] + 1 if i in selected_subset else self.count_each_task[i] for i in range(num_tasks)]
            selected_subset = list(range(num_tasks))
            if self.lookahead_mode:
                if self.total_count % self.lookahead_freq == 0:
                    collect_s = time.time()
                    if self.tag:
                        task_combs = (
                            list(combinations(list(range(num_tasks)), 1))
                        )

                    else:
                        task_combs = (
                            list(combinations(selected_subset, 1))
                            + list(combinations(selected_subset, 2))
                            + [
                                (i[0], i[0])
                                for i in list(combinations(selected_subset, 1))
                            ]
                        )
                    nolookahead_score = self.valiad_and_save_model(
                        self.evaluation_size, lookahead_eval=True
                    )
                    if self.rank == 0:
                        self.transfer_gain_collection["no_lookahead"].append(
                            nolookahead_score
                        )
                    for selected_tasks in task_combs:
                        update_model_parameters(self.model, self.backup_model)

                        # print("Lookahead task comb: ", selected_tasks)
                        (
                            lookahead_avg_score,
                            lookahead_avg_loss,
                        ) = self._train_one_batch_lookahead(batch_size, selected_tasks)
                        start = time.time()
                        lookahead_score = self.valiad_and_save_model(
                            self.evaluation_size, lookahead=True
                        )
                        if self.rank == 0:
                            # print(lookahead_score)
                            self.collect_time[tuple(selected_tasks)]["forward"].append(time.time()-start)
                            self.transfer_gain_collection[selected_tasks].append(
                                lookahead_score
                            )
                            # print('Begin MTL training')
                            # print(self.transfer_gain_collection)
                        # update_model_parameters(self.model, self.backup_model)
                        # self.collect_time.append(time.time()- collect_s)

            # update model parameters
            avg_score, avg_loss = self._train_one_batch(batch_size, selected_subset)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            episode += batch_size

        self.training_time.append(time.time() - s)
        self.valiad_and_save_model(self.evaluation_size)

        # Log Once, for each epoch
        self.logger.info(
            "Epoch {:3d}: Train ({:3.0f}%)  Score: {}  Loss: {}".format(
                epoch,
                100.0 * episode / train_num_episode,
                self.eval_res[-1].reshape(-1),
                loss_AM.avg,
            )
        )

        return score_AM.avg, loss_AM.avg

    def _train_one_batch_lookahead(self, batch_size, selected_tasks):
        # Prep
        ###############################################
        start = time.time()
        backup_model, backup_optimizer = self.setup_backup_model_opt(selected_tasks)
        num_tasks = len(selected_tasks)

        loss_mean_all = None
        score_mean_all = None

        loss_list = []
        backup_optimizer.zero_grad()
        # backup_model.zero_grad()
        for c in selected_tasks:
            # print("lookahead for task ", c)
            problem_idx, scale_id = self.select_env_cop(c)
            env = self.env_list[problem_idx][scale_id]
            problem = self.problem[problem_idx]
            env.load_problems(batch_size)
            reset_s, _, _ = env.reset()
            state, reward, done = env.pre_step()
            backup_model.pre_forward_oneCOP(reset_s, problem)
            loss_mean, score_mean = self.train_one_COP_loohahead(
                backup_model, env, problem, state, reward, done
            )
            if self.alg == "banditmtl":
                loss_list.append(self.weighted_method[c] * loss_mean)
            else:
                loss_list.append(loss_mean)
            # loss_mean.backward()
            if loss_mean_all is None:
                loss_mean_all = loss_mean
                score_mean_all = score_mean
            else:
                loss_mean_all += loss_mean
                score_mean_all += score_mean

        total_loss = torch.stack(loss_list)
        if self.alg == "naive":
            total_loss.mean().backward()
        elif self.alg == "banditmtl":
            total_loss.sum().backward()
        else:
            self.weighted_method.backward(
                losses=total_loss,
                shared_parameters=list(backup_model.shared_parameters()),
                task_specific_parameters=list(backup_model.task_specific_parameters()),
            )
        backward_end = time.time()
        if self.rank == 0:
            self.collect_time[tuple(selected_tasks)]["backward"].append(backward_end-start)
        backup_optimizer.step()
        if self.rank == 0:
            self.collect_time[tuple(selected_tasks)]["update"].append(time.time()-backward_end)

        if self.alg == "banditmtl":
            losses = total_loss.detach().cpu().numpy()
            q_k = [
                self.weighted_method[i] * math.exp(self.eta_p * losses[i])
                for i in range(num_tasks)
            ]
            lam = find_lambda(1e-15, 10, 2e5, 1e5, q_k, self.rho, num_tasks)
            q_lam = [math.pow(i, 1 / (lam + 1)) for i in q_k]
            q_sum = sum(q_lam)
            self.weighted_method = [i / q_sum for i in q_lam]
            self.weighted_collection.append(self.weighted_method)
        update_model_parameters(self.backup_model, backup_model, update_target=False)
        return loss_mean_all.data.item() / num_tasks, score_mean_all / num_tasks

    def train_one_COP_loohahead(self, backup_model, env, problem, state, reward, done):
        prob_list = torch.zeros(size=(env.batch_size, env.pomo_size, 0))

        # shape: (batch, pomo, 0~problem)
        while not done:
            selected, prob = backup_model(state, problem)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            try:
                # selected = state.true_selected  # the selected nodes may change due to some hard constrain, e.g. in OP
                prob = prob[state.BATCH_IDX, state.POMO_IDX, selected].reshape(
                    state.BATCH_IDX.size(0), state.BATCH_IDX.size(1)
                )
            except:
                pass
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = torch.abs(
            max_pomo_reward.float().mean()
        )  # negative sign to make positive value
        return loss_mean, score_mean.item()

    def virtual_step(self, alpha, model_optim, batch_size, selected_tasks):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        train_loss = []
        for c in selected_tasks:
            problem_idx, scale_id = self.select_env_cop(c)
            env = self.env_list[problem_idx][scale_id]
            problem = self.problem[problem_idx]
            env.load_problems(batch_size)
            reset_s, _, _ = env.reset()
            state, reward, done = env.pre_step()
            if self.ddp:
                self.model.module.pre_forward_oneCOP(reset_s, problem)
            else:
                self.model.pre_forward_oneCOP(reset_s, problem)
            loss_mean, score_mean = self.train_one_COP(
                env, problem, state, reward, done
            )
            train_loss.append(loss_mean)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.backup_model.parameters(), gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

    def unrolled_backward(self, alpha, model_optim, batch_size, selected_tasks):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(alpha, model_optim, batch_size, selected_tasks)

        # compute validation data loss on primary tasks
        val_loss = []
        for c in selected_tasks:
            problem_idx, scale_id = self.select_env_cop(c)
            env = self.env_list[problem_idx][scale_id]
            problem = self.problem[problem_idx]
            env.load_problems(batch_size)
            reset_s, _, _ = env.reset()
            state, reward, done = env.pre_step()
            if self.ddp:
                self.backup_model.module.pre_forward_oneCOP(reset_s, problem)
            else:
                self.backup_model.pre_forward_oneCOP(reset_s, problem)
            loss_mean, score_mean = self.train_one_COP_loohahead(
                self.backup_model, env, problem, state, reward, done
            )
            val_loss.append(loss_mean)

        loss = sum(val_loss)

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.backup_model.module.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, batch_size, selected_tasks)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = - alpha * h

    def compute_hessian(self, d_model, batch_size, selected_tasks):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d
        
        train_loss = []
        for c in selected_tasks:
            problem_idx, scale_id = self.select_env_cop(c)
            env = self.env_list[problem_idx][scale_id]
            problem = self.problem[problem_idx]
            env.load_problems(batch_size)
            reset_s, _, _ = env.reset()
            state, reward, done = env.pre_step()
            if self.ddp:
                self.model.module.pre_forward_oneCOP(reset_s, problem)
            else:
                self.model.pre_forward_oneCOP(reset_s, problem)
            loss_mean, score_mean = self.train_one_COP(
                env, problem, state, reward, done
            )
            train_loss.append(loss_mean)
        
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        train_loss = []
        for c in selected_tasks:
            problem_idx, scale_id = self.select_env_cop(c)
            env = self.env_list[problem_idx][scale_id]
            problem = self.problem[problem_idx]
            env.load_problems(batch_size)
            reset_s, _, _ = env.reset()
            state, reward, done = env.pre_step()
            if self.ddp:
                self.model.module.pre_forward_oneCOP(reset_s, problem)
            else:
                self.model.pre_forward_oneCOP(reset_s, problem)
            loss_mean, score_mean = self.train_one_COP(
                env, problem, state, reward, done
            )
            train_loss.append(loss_mean)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def _train_one_batch(self, batch_size, selected_tasks):
        # Prep
        ###############################################
        self.model.train()
        num_tasks = len(selected_tasks)
        if self.alg == "autol":
            self.meta_optimizer.zero_grad()
            self.unrolled_backward(self.scheduler.get_last_lr()[0], self.optimizer, batch_size, selected_tasks)
            self.meta_optimizer.step()
            with torch.no_grad():
                dist.all_reduce(self.meta_weights, op=dist.ReduceOp.SUM)
                self.meta_weights /= dist.get_world_size()
        loss_mean_all = None
        score_mean_all = None

        loss_list = []
        self.optimizer.zero_grad()
        for c in selected_tasks:
            problem_idx, scale_id = self.select_env_cop(c)
            env = self.env_list[problem_idx][scale_id]
            problem = self.problem[problem_idx]
            env.load_problems(batch_size)
            reset_s, _, _ = env.reset()
            state, reward, done = env.pre_step()
            if self.ddp:
                self.model.module.pre_forward_oneCOP(reset_s, problem)
            else:
                self.model.pre_forward_oneCOP(reset_s, problem)
            loss_mean, score_mean = self.train_one_COP(
                env, problem, state, reward, done
            )
            if self.alg == "banditmtl":
                loss_list.append(self.weighted_method[c] * loss_mean)
            else:
                loss_list.append(loss_mean)
            # loss_mean.backward()
            if loss_mean_all is None:
                loss_mean_all = loss_mean
                score_mean_all = score_mean
            else:
                loss_mean_all += loss_mean
                score_mean_all += score_mean

        total_loss = torch.stack(loss_list)
        if self.alg == "naive":
            total_loss.mean().backward()
        elif self.alg == "banditmtl":
            total_loss.sum().backward()
        elif self.alg == "autol":
            total_loss =  sum([w * loss_list[i] for i, w in enumerate(self.meta_weights)])
            total_loss.backward()
        else:
            self.weighted_method.backward(
                losses=total_loss,
                shared_parameters=list(get_inner_model(self.model.module).shared_parameters()),
                task_specific_parameters=list(
                    get_inner_model(self.model).task_specific_parameters()
                ),
            )
        self.optimizer.step()
        self.total_count += 1
        if self.alg == "banditmtl":
            losses = total_loss.detach().cpu().numpy()
            q_k = [
                self.weighted_method[i] * math.exp(self.eta_p * losses[i])
                for i in range(num_tasks)
            ]
            lam = find_lambda(1e-15, 10, 2e5, 1e5, q_k, self.rho, num_tasks)
            q_lam = [math.pow(i, 1 / (lam + 1)) for i in q_k]
            q_sum = sum(q_lam)
            self.weighted_method = [i / q_sum for i in q_lam]
            self.weighted_collection.append(self.weighted_method)
        return loss_mean_all.data.item() / num_tasks, score_mean_all / num_tasks

    def train_one_COP(self, env, problem, state, reward, done):
        prob_list = torch.zeros(size=(env.batch_size, env.pomo_size, 0))

        # shape: (batch, pomo, 0~problem)
        while not done:
            selected, prob = self.model(state, problem)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            try:
                # selected = state.true_selected  # the selected nodes may change due to some hard constrain, e.g. in OP
                prob = prob[state.BATCH_IDX, state.POMO_IDX, selected].reshape(
                    state.BATCH_IDX.size(0), state.BATCH_IDX.size(1)
                )
            except:
                pass
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = torch.abs(
            max_pomo_reward.float().mean()
        )  # negative sign to make positive value
        return loss_mean, score_mean.item()

    def valiadate(self, batch_size, lookahead=False):
        if lookahead:
            self.backup_model.eval()
        else:
            self.model.eval()

        def val_env_list(env_list, problems, batch_size, seen=False, fix_data=False):
            for i, cop_env in enumerate(env_list):
                for j, env in enumerate(cop_env):
                    if fix_data:
                        if seen:
                            validation_data = self.overall_seen_data[i][j]
                        else:
                            validation_data = self.overall_unseen_data[i][j]
                        env.load_problems(batch_size, prepare_dataset=validation_data)
                    else:
                        env.load_problems(batch_size)

            def val_one_model(model, params=None):
                if params is not None:
                    if self.ddp:
                        model.module.load_state_dict(params)
                    else:
                        model.load_state_dict(params)
                reset_state = []
                states, rewards, dones = [], [], []
                for cop_env in env_list:
                    temp_reset_state = []
                    temp_state = []
                    temp_reward = []
                    temp_dones = []
                    for env in cop_env:
                        reset_s, _, _ = env.reset()
                        state, reward, done = env.pre_step()
                        temp_reset_state.append(reset_s)
                        temp_state.append(state)
                        temp_reward.append(reward)
                        temp_dones.append(done)

                    reset_state.append(temp_reset_state)
                    states.append(temp_state)
                    rewards.append(temp_reward)
                    dones.append(temp_dones)

                score_list = []
                # states, rewards, dones = zip(*[env.pre_step() for env in self.env_list])
                for j in range(len(env_list)):
                    cop_env = env_list[j]
                    problem = problems[j]
                    # temp_score = []
                    for i in range(len(cop_env)):
                        env = cop_env[i]

                        with torch.no_grad():
                            if self.ddp:
                                model.module.pre_forward_oneCOP(reset_state[j][i], problem)
                            else:
                                model.pre_forward_oneCOP(reset_state[j][i], problem)
                            state, reward, done = (
                                states[j][i],
                                rewards[j][i],
                                dones[j][i],
                            )
                            # shape: (batch, pomo, 0~problem)
                            while not done:
                                selected, _ = model(state, problem)
                                # shape: (batch, pomo)
                                state, reward, done = env.step(selected)

                            # Score
                            ###############################################
                            max_pomo_reward, _ = reward.max(
                                dim=1
                            )  # get best results from pomo
                            score = torch.abs(
                                max_pomo_reward
                            )  # negative sign to make positive value
                            score_list.append(score.mean().view(-1))
                    # score_list.append(temp_score)
                try:
                    return torch.cat(score_list)
                except:
                    return torch.tensor([])

            if lookahead:
                res_current_model = val_one_model(self.backup_model)
            else:
                res_current_model = val_one_model(self.model)
            return res_current_model

        res_on_seen = val_env_list(
            self.env_list,
            self.problem,
            batch_size,
            seen=True,
            fix_data=True,
        )
        res_on_unseen = val_env_list(
            self.unseen_env_list,
            self.unseen_problem,
            batch_size,
            seen=False,
            fix_data=True,
        )
        return res_on_seen, res_on_unseen

    def valiad_and_save_model(self, batch_size, lookahead=False, lookahead_eval=False):
        if len(self.eval_res) != 0:
            eval_res_hist = np.concatenate(self.eval_res, axis=0)
        else:
            eval_res_hist = None
        cur_eval_res, unseen_eval_res = self.valiadate(batch_size, lookahead)
        total_res_mean = torch.cat([cur_eval_res, unseen_eval_res], dim=0)
        if self.ddp:
            dist.all_reduce(total_res_mean, op=dist.ReduceOp.SUM)
            total_res_mean /= dist.get_world_size()
        total_res_mean = total_res_mean.cpu().numpy()

        if not lookahead and not lookahead_eval:
            # update the historical best param on seen tasks
            if eval_res_hist is not None:
                temp_count = 0
                for i, cop_env in enumerate(self.env_list):
                    problem = self.problem[i]
                    for j, env in enumerate(cop_env):
                        if problem == "KP" or problem == "OP":
                            if total_res_mean[temp_count] > np.max(
                                eval_res_hist[:, temp_count]
                            ):
                                self.hist_best_model_params_seen[i][j][
                                    0
                                ] = self.total_count
                                if self.ddp:
                                    self.hist_best_model_params_seen[i][j][1] = deepcopy(
                                        self.model.module.state_dict()
                                    )
                                else:
                                    self.hist_best_model_params_seen[i][j][1] = deepcopy(
                                        self.model.state_dict()
                                    )
                        else:
                            if total_res_mean[temp_count] < np.min(
                                eval_res_hist[:, temp_count]
                            ):
                                self.hist_best_model_params_seen[i][j][
                                    0
                                ] = self.total_count
                                if self.ddp:
                                    self.hist_best_model_params_seen[i][j][1] = deepcopy(
                                        self.model.module.state_dict()
                                    )
                                else:
                                    self.hist_best_model_params_seen[i][j][1] = deepcopy(
                                    self.model.state_dict()
                                )

                        temp_count += 1

            self.eval_res.append(total_res_mean.reshape(1, -1))
        return total_res_mean.reshape(1, -1)

    def select_env_cop(self, choice):
        choice_id = choice + 1
        num_scales = np.array([len(cop_env) for cop_env in self.env_list])
        cum_sum = np.cumsum(num_scales)
        cop_id = np.where((cum_sum < choice_id) == False)[0][0]
        if cop_id == 0:
            scale_id = choice
        else:
            scale_id = choice - cum_sum[cop_id - 1]
        return cop_id, scale_id

    def setup_backup_model_opt(self, selected_task):
        # selected_problem = np.unique(
        #     [self.problem[self.select_env_cop(i)[0]] for i in selected_task]
        # ).tolist()
        backup_model = Model(self.problem, **self.model_params)

        update_model_parameters(self.model, backup_model)

        backup_optimizer = Optimizer(
            backup_model.parameters(), **self.optimizer_params["optimizer"]
        )
        # backup_optimizer.load_state_dict(self.optimizer.state_dict())

        update_opt_parameters(self.optimizer, backup_optimizer)

        return backup_model, backup_optimizer


def update_model_parameters(source_model, target_model, update_target=True):
    """
    Updates parameters of the target model with parameters from the source model.
    Only updates parameters that exist in both models.

    :param source_model: The model from which to copy parameters.
    :param target_model: The model to be updated.
    :param update_target: If True, update target_model from source_model.
                          If False, update source_model from target_model.
    :return: None
    """
    try:
        source_state_dict = source_model.module.state_dict()
    except:
        source_state_dict = source_model.state_dict()
    try:
        target_state_dict = target_model.module.state_dict()
    except:
        target_state_dict = target_model.state_dict()

    # Determine the direction of the update
    if update_target:
        # Update target model from source model
        # updated_state_dict = {key: source_state_dict[key] for key in target_state_dict}
        try:
            target_model.module.load_state_dict(source_state_dict)
        except:
            target_model.load_state_dict(source_state_dict)
    else:
        # Update source model from target model
        # updated_state_dict = {
        #     key: target_state_dict[key]
        #     if key in target_state_dict
        #     else source_state_dict[key]
        #     for key in source_state_dict
        # }
        try:
            source_model.module.load_state_dict(target_state_dict)
        except:
            source_model.load_state_dict(target_state_dict)


def update_opt_parameters(source_opt, target_opt):
    # Step 1: Extract relevant parameters from opt_all's state_dict
    opt_all_state_dict = source_opt.state_dict()
    opt_sub_state_dict = target_opt.state_dict()

    # Assuming that model_sub's parameters are a subset of model_all's,
    # filter out only the relevant state for opt_sub
    # for key in opt_sub_state_dict["state"].keys():
    #     if key in opt_all_state_dict["state"]:
    #         opt_sub_state_dict["state"][key] = opt_all_state_dict["state"][key]

    # Step 2: Load the filtered state_dict into opt_sub
    target_opt.load_state_dict(opt_all_state_dict)
