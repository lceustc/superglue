export GLUE_DIR=/data/lce/pytorch-transformers/data
export TASK_NAME=CB

for SEED in   3 7 42 50 87 99
do

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
    --model_type roberta\
    --model_name_or_path /data/lce/pretrained_model/roberta-mnli/ \
    --task_name=$TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=8   \
    --learning_rate=1e-5 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=10 \
    --logging_steps=-15  \
    --warmup=0.1 \
    --save_steps=-1 \
    --output_dir /data/lce/models/$TASK_NAME/cb_mnli_mixout/$SEED/ \
    --seed=$SEED \
    --is_mixout \

done
