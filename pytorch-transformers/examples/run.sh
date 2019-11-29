export GLUE_DIR=/data/lce/pytorch-transformers/data/test
export TASK_NAME=pretrain

for SEED in     42
do

CUDA_VISIBLE_DEVICES=6 python run_map.py \
    --model_type bert \
    --model_name_or_path /data/lce/pretrained_model/bert-base \
    --model_name_or_path_cn /data/lce/pretrained_model/bert-cn \
    --task_name=$TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --data_dir /data/lce/data/ \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=8   \
    --learning_rate=1e-5 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=10 \
    --logging_steps=50  \
    --save_steps=-1 \
    --output_dir ./pretrain \
    --seed=$SEED \

done
