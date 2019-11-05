for SEEDS in 3 7 42 50 87

do
CUDA_VISIBLE_DEVICES=7 python run_ad.py \
        --data_dir '/data/lce/pytorch-transformers/data/RTE' \
        --model_type 'roberta' \
        --model_name_or_path '/data/lce/pytorch-transformers/examples/pretrained_model/roberta-mnli2' \
        --ignore_logits_layer \
        --task_name 'rte' \
        --output_dir "./rte/" \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --evaluate_during_training \
        --per_gpu_train_batch_size 8 \
        --overwrite_output_dir \
        --per_gpu_eval_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-5 \
        --warmup_proportion 0.1 \
        --num_train_epochs 10 \
        --logging_steps 100 \
        --save_steps -1 \
        --seed $SEEDS \
        --ema -1 \
        --adversarial_training_weight 1 \
        --adversarial_training_eps 1
done