export GLUE_DIR=/data/lce/pytorch-transformers/data
export TASK_NAME=COPA

for SEED in 3 7 42 50 99
do

CUDA_VISIBLE_DEVICES=7 python cv_cb.py \
    --model_type roberta_mc\
    --model_name_or_path /data/lce/pretrained_model/roberta-mnli \
    --task_name=$TASK_NAME \
    --evaluate_during_training \
    --do_train \
    --data_dir $GLUE_DIR/$TASK_NAME/ \
    --max_seq_length 32 \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4   \
    --learning_rate=1e-5 \
    --warmup=0.1 \
    --overwrite_output_dir \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=10 \
    --logging_steps=1  \
    --save_steps=1 \
    --output_dir /data/lce/models/$TASK_NAME/mnli_eda_at/$SEED \
    --seed=$SEED \
    --lstm_ad=1 \

done
