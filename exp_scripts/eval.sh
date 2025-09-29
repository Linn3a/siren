#!/bin/bash

cd eval

export VLLM_ATTENTION_BACKEND=XFORMERS


GPUS=
OUTPUT=.

datasets=(
all
)

models=(
    /mnt/shared-storage-user/jiangyuxian/models/Qwen2.5-Math-1.5B
)

temperatures=(
    0.6
) 


for path in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for tem in "${temperatures[@]}"; do
            echo "[Currently] $path"
            echo "[Currently] $dataset"
            echo "[Currently] $tem $temperature"
            start_time=$(date +%s)
            python3 eval_entropy.py \
            --model_path=${path} \
            --dataset=${dataset} \
            --temperature=${tem} \
            --gpus=${GPUS}
            end_time=$(date +%s)
            elapsed_time=$((end_time - start_time))
            echo "${path},${dataset} time elapsed: $elapsed_time seconds"
            echo "------------------------------------------------------------"
            echo "------------------------------------------------------------"
        done
    done
done
