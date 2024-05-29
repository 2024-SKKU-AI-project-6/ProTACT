#!/usr/bin/env bash
model_name='ProTACT'

for prompt in 1
do
    python train_ProTACT.py --test_prompt_id ${prompt} --model_name ${model_name} --seed 1 --num_heads 2 --features_path './../data/LDA/hand_crafted_final_'
done