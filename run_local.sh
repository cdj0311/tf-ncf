#!/bin/sh
ckpt_dir=./ckpt
model_dir=./model

train_data=train_files.txt
eval_data=eval_files.txt
user_model_path="./user_model/"
item_model_path="./item_model/"
train_steps=10000000
batch_size=256
learning_rate=0.001
save_steps=100000
userID_bucket=20000000
itemID_bucket=1000000
embed_size=64

python basic.py \
    --train_data=${train_data} \
    --eval_data=${eval_data} \
    --model_dir=${ckpt_dir} \
    --output_model=${model_dir} \
    --train_steps=${train_steps} \
    --save_checkpoints_steps=${save_steps} \
    --learning_rate=${learning_rate} \
    --batch_size=${batch_size} \
    --userID_size=${userID_bucket} \
    --itemID_size=${itemID_bucket} \
    --embed_size=${embed_size} \
    --user_model_path=${user_model_path} \
    --item_model_path=${item_model_path} \
    --is_eval=False \
    --run_on_cluster=False \
    --export_user_model=True \
    --export_item_model=True \
    --train_eval_model=True \
    --gpuid=1

