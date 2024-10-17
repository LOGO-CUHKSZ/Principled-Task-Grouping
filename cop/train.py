import os
import torch
import logging
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
sys.path.insert(0, "..")

from Trainer import Trainer
from utils import create_logger, copy_all_src

import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
import socket

warnings.filterwarnings("ignore")


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _, port = s.getsockname()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return port


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _print_config(trainer_params):
    logger = logging.getLogger("root")
    logger.info(
        "USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(
            trainer_params["use_cuda"], trainer_params["cuda_device_num"]
        )
    )
    [
        logger.info(g_key + "{}".format(globals()[g_key]))
        for g_key in globals().keys()
        if g_key.endswith("params")
    ]


def ddp_train(
    rank,
    world_size,
    env_params,
    model_params,
    trainer_params,
    optimizer_params,
    logger_params,
    opts,
):
    print(f"DDP training on rank {rank}.")
    setup(rank, world_size)
    main(
        rank,
        opts,
        env_params,
        model_params,
        trainer_params,
        optimizer_params,
        logger_params,
    )
    cleanup()


def main(
    rank,
    opts,
    env_params,
    model_params,
    trainer_params,
    optimizer_params,
    logger_params,
):
    if rank == 0:
        create_logger(**logger_params)
        _print_config(trainer_params)

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
        rank=rank,
        opts=opts,
    )
    if rank == 0:
        copy_all_src(trainer.result_folder)
    trainer.run()


if __name__ == "__main__":
    from options import get_options
    import yaml

    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    env_params = config["env_params"]
    unseen_env_params = config["unseen_env_params"]

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]
    optimizer_params = config["optimizer_params"]
    logger_params = config["logger_params"]

    model_params["sqrt_embedding_dim"] = model_params["embedding_dim"] ** (0.5)

    opts = get_options()
    # CUDA_VISIBLE_DEVICES=6,7 python train.py --alg banditmtl --tsp 20 50 --cvrp 20 50 --op 20 50 --kp 50 100 --epochs 100 --task_description test_small --model_save_interval 10

    # os.environ["CUDA_VISIBLE_DEVICES"] = "4, 6"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # opts.hfai_mode = True
    # opts.alg = "naive"
    # opts.tsp = [20, 50]
    # opts.cvrp = [20, 50]
    # opts.op = [20, 50]
    # opts.kp = [50]
    #
    # opts.unseen_tsp = [21]
    # opts.unseen_cvrp = [21]
    # opts.unseen_op = [21]
    # opts.unseen_kp = [51]
    #

    # opts.train_episodes = 1000
    # opts.train_batch_size = 128
    # opts.task_description = 'test-valid-nogap-debug---'

    # if opts.select_freq is None:
    #     opts.select_freq = opts.train_episodes//opts.train_batch_size if opts.train_episodes%opts.train_batch_size==0 else opts.train_episodes//opts.train_batch_size + 1

    # setting for seen tasks
    if opts.tsp is not None:
        env_params["TSP"]["problem_size"] = opts.tsp
        env_params["TSP"]["pomo_size"] = [min(s, 100) for s in opts.tsp]
    if opts.cvrp is not None:
        env_params["CVRP"]["problem_size"] = opts.cvrp
        env_params["CVRP"]["pomo_size"] = [min(s, 100) for s in opts.cvrp]
    if opts.op is not None:
        env_params["OP"]["problem_size"] = opts.op
        env_params["OP"]["pomo_size"] = [min(s, 100) for s in opts.op]
    if opts.kp is not None:
        env_params["KP"]["problem_size"] = opts.kp
        env_params["KP"]["pomo_size"] = [min(s, 100) for s in opts.kp]
    new_env_params = {}
    new_env_params["same"] = opts.coord_same

    for key in env_params.keys():
        try:
            if env_params[key]["problem_size"] != None:
                new_env_params[key] = env_params[key]
        except:
            pass
    env_params = new_env_params
    problem_list = list(env_params.keys())
    problem_list.remove("same")

    # settings for unseen tasks
    if opts.unseen_tsp is not None:
        unseen_env_params["TSP"]["problem_size"] = opts.unseen_tsp
        unseen_env_params["TSP"]["pomo_size"] = [min(s, 100) for s in opts.unseen_tsp]
    if opts.unseen_cvrp is not None:
        unseen_env_params["CVRP"]["problem_size"] = opts.unseen_cvrp
        unseen_env_params["CVRP"]["pomo_size"] = [min(s, 100) for s in opts.unseen_cvrp]
    if opts.unseen_op is not None:
        unseen_env_params["OP"]["problem_size"] = opts.unseen_op
        unseen_env_params["OP"]["pomo_size"] = [min(s, 100) for s in opts.unseen_op]
    if opts.unseen_kp is not None:
        unseen_env_params["KP"]["problem_size"] = opts.unseen_kp
        unseen_env_params["KP"]["pomo_size"] = [min(s, 100) for s in opts.unseen_kp]
    new_unseen_env_params = {}

    for key in unseen_env_params.keys():
        try:
            if unseen_env_params[key]["problem_size"] != None:
                new_unseen_env_params[key] = unseen_env_params[key]
        except:
            pass
    unseen_env_params = new_unseen_env_params
    unseen_problem_list = list(unseen_env_params.keys())

    trainer_params["separate_train"] = opts.separate_train
    trainer_params["epochs"] = opts.epochs
    trainer_params["rew_alpha"] = opts.rew_alpha

    if trainer_params["cusumucb"]["max_nb_random_events"] is None:
        trainer_params["cusumucb"]["max_nb_random_events"] = opts.epochs / (
            2 * (1 - 0.1**2)
        )  # we set alpha=0.1

    trainer_params["logging"]["model_save_interval"] = opts.model_save_interval
    if opts.model_load:
        trainer_params["model_load"]["enable"] = True
        assert opts.resume_path is not None and opts.resume_epoch is not None
        trainer_params["model_load"]["path"] = opts.resume_path
        trainer_params["model_load"]["epoch"] = opts.resume_epoch

    if opts.epochs % 100 == 0:
        milestone = (opts.epochs // 100 - 1) * 100
    else:
        milestone = opts.epochs // 100 * 100
    optimizer_params["scheduler"]["milestones"] = [
        900,
        2900,
        4900,
    ]  # if (not trainer_params['separate_train']) or (opts.bandit_alg is not None) else [milestone * len(problem_list),]
    logger_params["log_file"]["desc"] = "train_{}_alg-{}_desc-{}".format(
        "-".join(str(_) + str(env_params[_]["problem_size"]) for _ in problem_list),
        opts.alg,
        opts.task_description,
    )

    if opts.hfai_mode:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'#'0,1,2,3,4,5,6,7'
        resume_path = "./result/_train_{}_alg-{}_desc-{}".format(
            "-".join(str(_) + str(env_params[_]["problem_size"]) for _ in problem_list),
            opts.alg,
            opts.task_description,
        )
        try:
            if "checkpoint-latest.pt" in os.listdir(resume_path):
                trainer_params["model_load"] = {
                    "enable": True,  # enable loading pre-trained model
                    "path": resume_path,  # directory path of pre-trained model and log files saved.
                    "epoch": "latest",  # epoch version of pre-trained model to laod.
                }
                print("Resume Training!")
        except:
            pass

        logger_params["log_file"]["filepath"] = "./result/" + "{desc}"

    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    assert (
        n_gpus >= world_size
    ), f"Requires at least {world_size} GPUs to run, but got {n_gpus}"
    print(f"Using {world_size} GPUs.")
    trainer_params["train_episodes"] = opts.train_episodes // world_size
    trainer_params["train_batch_size"] = opts.train_batch_size
    opts.evaluation_size = (
        opts.evaluation_size // world_size
        if opts.evaluation_size % world_size == 0
        else opts.evaluation_size // world_size + 1
    )

    total_env_prams = {"seen": env_params, "unseen": unseen_env_params}
    # check to number of gpus used
    if world_size == 1:
        opts.ddp = False
        main(
        0,
        opts,
        total_env_prams,
        model_params,
        trainer_params,
        optimizer_params,
        logger_params,
    )
        
    else:
        opts.ddp = True
        os.environ["MASTER_PORT"] = str(find_available_port())
        mp.spawn(
            ddp_train,
            args=(
                world_size,
                total_env_prams,
                model_params,
                trainer_params,
                optimizer_params,
                logger_params,
                opts,
            ),
            nprocs=world_size,
            join=True,
        )
