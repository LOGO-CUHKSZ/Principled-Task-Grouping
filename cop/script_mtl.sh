CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500 --tsp 20 50 --cvrp 20 50 --op 20 50  --alg pcgrad --task_description 'pcgrad_comb' --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500 --tsp 20 50 --cvrp 20 50 --op 20 50  --alg nashmtl --task_description 'nashmtl_comb' --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500 --tsp 20 50 --cvrp 20 50 --op 20 50  --alg banditmtl --task_description 'banditmtl_comb' --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500 --tsp 20 50 --cvrp 20 50 --op 20 50  --alg cagrad --task_description 'cagrad_comb' --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500 --tsp 20 50 --cvrp 20 50 --op 20 50  --alg ls --task_description 'ls_comb' --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500 --tsp 20 50 --cvrp 20 50 --op 20 50  --alg autol --task_description 'autol_comb' --model_save_interval 10
