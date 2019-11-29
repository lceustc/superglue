export GLUE_DIR=/data/lce/pytorch-transformers/data
export TASK_NAME=RTE

for SEED in   3
do

CUDA_VISIBLE_DEVICES=1 python cv_cb.py \
    --model_type roberta\
    --model_name_or_path /data/lce/pretrained_model/roberta-mnli \
    --task_name=$TASK_NAME \
    --evaluate_during_training \
    --do_predict \
    --data_dir $GLUE_DIR/$TASK_NAME/ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4  \
    --learning_rate=2e-5 \
    --gradient_accumulation_steps=4 \
    --warmup=0.06 \
    --num_train_epochs=13 \
    --logging_steps=-1  \
    --save_steps=1 \
    --output_dir /data/lce/models/$TASK_NAME/cv_freeLb_mnli/$SEED \
    --seed=$SEED \
    --freeLB=1 \

done
