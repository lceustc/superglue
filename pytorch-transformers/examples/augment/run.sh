python augment.py \
    --data '/root/GLUE/data-superglue/BoolQ/train.jsonl' \
    --task 'boolq' \
    --exp_name 'synonym_replacement_4_0.2' \
    --do_synonym_replacement \
    --num_aug 4 \
    --alpha 0.2 \
    --aug_passage \
