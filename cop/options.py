import argparse


def get_options():
    parser = argparse.ArgumentParser()
    # lookahead
    parser.add_argument("--lookahead_mode", action="store_true")
    parser.add_argument(
        "--lookahead_freq",
        type=int,
        default=10,
        help="Frequency of collecting lookahead information",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default=None,
        help="use sampling mode for accelerating the collection",
    )
    parser.add_argument("--tag", action="store_true")

    parser.add_argument("--hfai_mode", action="store_true")
    parser.add_argument(
        "--alg", default=None, help="naive, pcgrad, nashmtl, banditmtl, uw"
    )
    parser.add_argument("--method_params_lr", type=float, default=0.025)

    # NashMTL
    parser.add_argument(
        "--nashmtl_optim_niter", type=int, default=20, help="number of CCCP iterations"
    )
    parser.add_argument(
        "--update_weights_every",
        type=int,
        default=1,
        help="update task weights every x iterations.",
    )
    # stl
    parser.add_argument(
        "--main-task",
        type=int,
        default=0,
        help="main task for stl. Ignored if method != stl",
    )

    # cagrad
    parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")
    # dwa
    # dwa
    parser.add_argument(
        "--dwa_temp",
        type=float,
        default=2.0,
        help="Temperature hyper-parameter for DWA. Default to 2 like in the original paper.",
    )
    # banditmtl
    parser.add_argument("--rho", type=float, default=1.2)
    parser.add_argument("--eta_p", type=float, default=0.5)
    # Auto-lambda
    parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
    parser.add_argument('--autol_lr', default=1e-4, type=float, help='learning rate for auto-lambda')
    # problem setting
    # seen tasks
    parser.add_argument("--tsp", nargs="+", type=int, default=None)
    parser.add_argument("--cvrp", nargs="+", type=int, default=None)
    parser.add_argument("--op", nargs="+", type=int, default=None)
    parser.add_argument("--kp", nargs="+", type=int, default=None)
    # unseen tasks
    parser.add_argument("--unseen_tsp", nargs="+", type=int, default=None)
    parser.add_argument("--unseen_cvrp", nargs="+", type=int, default=None)
    parser.add_argument("--unseen_op", nargs="+", type=int, default=None)
    parser.add_argument("--unseen_kp", nargs="+", type=int, default=None)
    # training mode
    parser.add_argument("--coord_same", action="store_true")
    parser.add_argument("--separate_train", type=bool, default=False)
    parser.add_argument("--rew_alpha", type=float, default=0.5)

    # training params
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train_episodes", type=int, default=10 * 1000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-8)

    parser.add_argument("--evaluation_size", type=int, default=512)
    parser.add_argument("--model_save_interval", type=int, default=50)
    parser.add_argument("--model_load", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_epoch", default=None)

    parser.add_argument("--task_description", type=str, default=None)

    parser.add_argument("--dynamic_info", type=str, default='dynamic_info-cop.pkl')
    parser.add_argument("--grouping", type=str, default=None)

    parser.add_argument("--decoder_path", type=str, default='mtl-checkpoint.pt')
    parser.add_argument("--load_opt", action="store_true")
    parser.add_argument("--opt_offset", type=int, default=0)

    opts = parser.parse_args()
    return opts
