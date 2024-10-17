accelerate launch --config_file=acc_config.yaml --main_process_port 30501 ours_train.py \
    --tasks sdnkt \
    --arch xception_taskonomy_new \
    --data_dir /data2/taskonomy_dataset \
    -r -b 24 \
    --epochs 10 \
    --model_dir 'saved_models/collect'