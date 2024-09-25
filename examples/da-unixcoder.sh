python llm2binfuncsim/train_bash.py \
    --stage da \
    --model_name_or_path microsoft/unixcoder-base \
    --dataset_dir ./examples/ \
    --cutoff_len 512 \
    --subsampling_probs 0.001,0.001,0.0005 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --cache_dir ./model_cache \
    --overwrite_cache \
    --output_dir ./unixcoder-base/ \
    --overwrite_output_dir \
    --logging_steps 1 \
    --save_strategy "no" \
    --save_total_limit 5 \
    --use_auth_token True \
    --use_custom_callback \
    --call_back_save_epochs 3 \
    --remove_unused_columns true \
    --load_best_model_at_end true 