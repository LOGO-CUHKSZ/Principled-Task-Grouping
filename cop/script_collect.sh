CUDA_VISIBLE_DEVICES=2 python train.py --lookahead_mode --lookahead_freq 1 --epochs 10 --tsp 20 50 --cvrp 20 50 --op 20 50 --alg naive --train_batch_size 400  --train_episodes 10000 --task_description 'tg_tsp_cvrp_op_collect_ours' --model_save_interval 5
