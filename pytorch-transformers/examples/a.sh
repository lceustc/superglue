export GLUE_DIR=/data/lce/pytorch-transformers/data
export TASK_NAME=RTE

for SEED in  3 7 42 50 87 99 107
do

CUDA_VISIBLE_DEVICES=4 python cv_cb.py \
    --model_type roberta\
    --model_name_or_path /data/lce/pretrained_model/roberta-mnli \
    --task_name=$TASK_NAME \
    --evaluate_during_training \
    --do_train \
    --data_dir $GLUE_DIR/$TASK_NAME/RTE-eda_9-alpha_.1-premise \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=8  \
    --learning_rate=1e-5 \
    --warmup=0.1 \
    --overwrite_output_dir \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=10 \
    --logging_steps=-1  \
    --save_steps=-1 \
    --output_dir ./tmp \
    --seed=$SEED \
    --lstm_ad=1 \

done
