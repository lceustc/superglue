export GLUE_DIR=/data/lce/pytorch-transformers/data
export TASK_NAME=COPA

for SEED in   3 7 42 50 87 99
do

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
    --model_type roberta_mc\
    --model_name_or_path /data/lce/pytorch-transformers/examples/pretrained_model/roberta-swag \
    --task_name=$TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4   \
    --learning_rate=1e-5 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=10 \
    --logging_steps=15  \
    --warmup=0.1 \
    --save_steps=-1 \
    --output_dir ./models/tmp \
    --seed=$SEED \

done
