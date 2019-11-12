export GLUE_DIR=/data/lce/pytorch-transformers/data
export TASK_NAME=RTE

for SEED in   3
do

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
    --model_type roberta\
    --model_name_or_path /data/lce/pretrained_model/roberta-mnli/ \
    --task_name=$TASK_NAME \
    --do_eval \
    --evaluate_during_training \
    --data_dir $GLUE_DIR/$TASK_NAME/ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4   \
    --learning_rate=1e-5 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=10 \
    --logging_steps=63 \
    --warmup=0.06 \
    --badcase \
    --save_steps=-1 \
    --output_dir /data/lce/models/$TASK_NAME/mnli_cv_at/$SEED/checkpoint-3-0-2 \
    --seed=$SEED \

done
