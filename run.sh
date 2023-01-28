#!/bin/bash

# BERT-base
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/afnc-base_phi07_top3_str1_adelim \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --temp 0.05 \
    --topk 3 \
    --phi 0.7 \
    --screen_strategy Strategy_1 \
    --loss_strategy adaptive_elimination \
    --do_train \
    --do_eval \
    --fp16



# BERT-large
python train.py \
   --model_name_or_path bert-large-uncased \
   --train_file data/wiki1m_for_simcse.txt \
   --output_dir result/afnc-large_phi08_top1_str1_adelim\
   --num_train_epochs 1 \
   --per_device_train_batch_size 64 \
   --learning_rate 1e-5 \
   --max_seq_length 32 \
   --evaluation_strategy steps \
   --metric_for_best_model stsb_spearman \
   --load_best_model_at_end \
   --eval_steps 125 \
   --pooler_type cls \
   --mlp_only_train \
   --temp 0.05 \
   --topk 1 \
   --phi 0.8 \
   --screen_strategy Strategy_1 \
   --loss_strategy adaptive_elimination \
   --do_train \
   --do_eval \
   --fp16





