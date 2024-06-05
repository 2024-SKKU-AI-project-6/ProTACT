#!/usr/bin/env bash
model_name='ProTACT'

# for prompt in 1 2 3 4 5 6 7 8
for epochs in 50
do
    python train_ProTACT.py --test_prompt_id 1 --model_name ${model_name} --seed 1 --num_heads 2 --features_path './../data/LDA/hand_crafted_final_' --epochs ${epochs}
done