CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500  --tsp 20  --alg naive --task_description "stl_naive" --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500  --tsp 50  --alg naive --task_description "stl_naive" --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500  --cvrp 20  --alg naive --task_description "stl_naive" --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500  --cvrp 50  --alg naive --task_description "stl_naive" --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500  --op 20  --alg naive --task_description "stl_naive" --model_save_interval 10
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --epochs 500  --op 50  --alg naive --task_description "stl_naive" --model_save_interval 10