#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m llava.eval_visiondrop.model_vqa_science \
    --model-path /data0_1/MLLM/liuhaotian/llava-v1.5-7b \
    --question-file /data0_1/MLLM/scienceqa/llava_test_CQM-A.json \
    --image-folder /data0_1/MLLM/scienceqa/images/test \
    --answers-file /data/MLLM/scienceqa/answers/llava-v1.5-7b_visiondrop.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list [8,16,24] \
    --image_token_list [[30,5],[22,4],[16,3]] \
    --dominant 42 \
    --contextual 6 \
    --conv-mode vicuna_v1

python llava/eval_visiondrop/eval_science_qa.py \
    --base-dir /data0_1/MLLM/scienceqa \
    --result-file /data/MLLM/scienceqa/answers/llava-v1.5-7b_visiondrop.jsonl \
    --output-file /data/MLLM/scienceqa/answers/llava-v1.5-7b_visiondrop_output.jsonl \
    --output-result /data/MLLM/scienceqa/answers/llava-v1.5-7b_visiondrop_result.json
