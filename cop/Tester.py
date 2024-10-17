import torch
import os
from logging import getLogger
from Env.TSPEnv import TSPEnv
from Env.CVRPEnv import CVRPEnv
from Env.COPEnv import COPEnv as Env
from Models.models import COPModel as Model
from utils import *
import pickle


class COPTester:
    def __init__(self,
                 test_env_params,
                 model_params,
                 tester_params,
                 ):

        model_load = tester_params['model_load']

        args_file = '{path}/args.json'.format(**model_load)
        with open(args_file, 'r') as f:
            args = json.load(f)

        self.env_params = {}
        try:
            if args['tsp'] is not None:
                self.env_params['TSP'] = args['tsp']

            if args['cvrp'] is not None:
                self.env_params['CVRP'] = args['cvrp']

            if args['op'] is not None:
                self.env_params['OP'] = args['op']

            if args['kp'] is not None:
                self.env_params['KP'] = args['kp']
        except:
            if 'tsp' in args:
                self.env_params['TSP'] = args['tsp']

            if 'cvrp' in args:
                self.env_params['CVRP'] = args['cvrp']

            if 'op' in args:
                self.env_params['OP'] = args['op']

            if 'kp' in args:
                self.env_params['KP'] = args['kp']

        self.test_env_params = test_env_params
        self.model_params = model_params
        try:
            self.model_params['encoder_layer_num'] = args['encoder_layer_num']
        except:
            pass
        self.tester_params = tester_params
        self.problem = list(self.env_params.keys())
        self.test_problem = list(self.test_env_params.keys())

        self.test_data = []
        self.gt = []
        for problem in self.test_problem:
            self.test_data.append([])
            self.gt.append([])
            scales = self.test_env_params[problem]['problem_size']
            for scale in scales:
                with open('./datasets/{}/validation/{}-{}-10000.pkl'.format(problem, problem, scale), 'rb') as f:
                    data_gt = pickle.load(f)
                    self.test_data[-1].append(data_gt['data'].float().cuda())
                    self.gt[-1].append(data_gt['gt'])

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        if os.path.exists(self.result_folder+'/result_gap_epoch500.pkl'):
            self.stop = True
            print("exists!")
        else:
            self.stop = False

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env_list = Env(**self.test_env_params).env_list
        self.model = Model(self.problem,**self.model_params)

        # Restore
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)

        try:
            seen_params = checkpoint['hist_best_model_params_seen'][1]
            unseen_params = checkpoint['hist_best_model_params_unseen'][1]
            self.best_param = [seen_params[i] + unseen_params[i] for i in range(len(seen_params))]
        except:
            try:
                seen_params = checkpoint['hist_best_model_params_seen']
                unseen_params = checkpoint['hist_best_model_params_unseen']
                self.best_param = [seen_params[i][1] + unseen_params[i][1] for i in range(len(seen_params))]
            except:
                try:
                    seen_params, unseen_params = [], []
                    for cop_param in checkpoint['hist_best_model_params_seen']:
                        seen_params.append([])
                        for param in cop_param:
                            seen_params[-1].append(param[1])
                    for cop_param in checkpoint['hist_best_model_params_unseen']:
                        unseen_params.append([])
                        for param in cop_param:
                            unseen_params[-1].append(param[1])
                    self.best_param = [seen_params[i] + unseen_params[i] for i in range(len(seen_params))]
                except:
                    try:
                        seen_params = checkpoint['hist_best_model_params_seen']
                        self.best_param = [[param[1] for param in seen_params[i]] for i in range(len(seen_params))]
                    except:
                        pass

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.model_epoch = model_load['epoch']

    def run(self, best_mode=False):
        if not self.stop:
        
            self.time_estimator.reset()

            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()

            test_num_episode = self.tester_params['test_episodes']
            episode = 0

            while episode < test_num_episode:

                remaining = test_num_episode - episode
                batch_size = min(self.tester_params['test_batch_size'], remaining)

                score_list, aug_score_list = self._test_one_batch(batch_size,episode,best_mode)
                gap_list = []
                aug_gap_list = []
                for i, cop_score in enumerate(score_list):
                    problem = self.test_problem[i]
                    for j, score in enumerate(cop_score):
                        aug_score =  aug_score_list[i][j]
                        gt = self.gt[i][j][episode:episode+batch_size]
                        if problem == 'KP' or problem == 'OP':
                            gap = (1 - score/gt).mean().item()*100
                            aug_gap = (1-aug_score/gt).mean().item()*100
                        else:
                            gap = (score/gt-1).mean().item()*100
                            aug_gap = (aug_score/gt-1).mean().item()*100
                        gap_list.append(gap)
                        aug_gap_list.append(aug_gap)


                score_AM.update(np.array(gap_list), batch_size)
                aug_score_AM.update(np.array(aug_gap_list), batch_size)

                episode += batch_size

                ############################
                # Logs
                ############################
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
                self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], gap:{}%, aug_gap:{}%".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, gap_list, aug_gap_list))

                all_done = (episode == test_num_episode)

                if all_done:
                    # self.logger.info(" *** Test Done *** ")
                    # self.logger.info(" NO-AUG SCORE: {} ".format(score_AM.avg))
                    # self.logger.info(" AUGMENTATION SCORE: {} ".format(aug_score_AM.avg))
                    result = {'no_aug_gap':score_AM.avg,
                            'aug_gap':aug_score_AM.avg}
                    if best_mode:
                        with open('{}/result_gap_epoch{}-best.pkl'.format(self.result_folder,self.model_epoch), 'wb') as file:
                            pickle.dump(result, file)
                    else:
                        with open('{}/result_gap_epoch{}.pkl'.format(self.result_folder,self.model_epoch), 'wb') as file:
                            pickle.dump(result, file)




    def _test_one_batch(self, batch_size,episode,best_mode=False):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        for i,cop_env in enumerate(self.env_list):
            cop_data = self.test_data[i]
            if self.test_problem[i] != 'KP':
                for j,env in enumerate(cop_env):
                    env.load_problems(batch_size,aug_factor,cop_data[j][episode:episode+batch_size])
            else:
                for j,env in enumerate(cop_env):
                    env.load_problems(batch_size,prepare_dataset=cop_data[j][episode:episode+batch_size])
        # set same coordinates for all COP

        # reset_state, _, _ = zip(*[env.reset() for env in self.env_list])
        reset_state = []
        states, rewards, dones = [], [], []
        for cop_env in self.env_list:
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

        no_aug_score_list, aug_score_list = [], []
        with torch.no_grad():
            for k in range(len(self.env_list)):
                no_aug_score_list.append([])
                aug_score_list.append([])
                cop_env = self.env_list[k]
                problem = self.test_problem[k]

                for i in range(len(cop_env)):
                    env = cop_env[i]
                    state, reward, done = states[k][i], rewards[k][i], dones[k][i]
                    if best_mode:
                        self.model.load_state_dict(self.best_param[k][i])
                    self.model.pre_forward_oneCOP(reset_state[k][i], problem)
                    while not done:
                        selected, _ = self.model(state, problem)
                        # shape: (batch, pomo)
                        state, reward, done = env.step(selected)


                    # Return
                    ###############################################
                    if problem == 'KP':
                        aug_factor = 1

                    if problem == 'OP':
                        a = 1

                    aug_reward = reward.reshape(aug_factor, env.batch_size//aug_factor, env.pomo_size)
                    # shape: (augmentation, batch, pomo)

                    max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
                    # shape: (augmentation, batch)
                    no_aug_score = torch.abs(max_pomo_reward[0, :].float())  # negative sign to make positive value

                    max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
                    # shape: (batch,)
                    aug_score = torch.abs(max_aug_pomo_reward.float())  # negative sign to make positive value

                    no_aug_score_list[-1].append(no_aug_score.cpu().numpy())
                    aug_score_list[-1].append(aug_score.cpu().numpy())
        return no_aug_score_list, aug_score_list

    def get_atten_weights(self):
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        atten_weights = []
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            atten_weights_one_batch = self.get_atten_weights_one_batch(batch_size)
            atten_weights.append(atten_weights_one_batch)
            episode += batch_size
        return torch.cat(atten_weights,dim=0)

    def get_atten_weights_one_batch(self, batch_size):
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        # Ready
        ###############################################
        self.model.eval()
        [env.load_problems(batch_size, aug_factor) for env in self.env_list]
        self.env_list[0].problems = self.env_list[1].depot_node_xy[:, 1:, :]
        reset_state, _, _ = zip(*[env.reset() for env in self.env_list])
        with torch.no_grad():
            atten_weights = self.model.get_atten_weights(reset_state)
        return atten_weights
