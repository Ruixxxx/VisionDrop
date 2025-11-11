#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m llava.eval_visiondrop2.model_vqa_loader \
    --model-path /data0_1/MLLM/liuhaotian/llava-v1.5-7b \
    --question-file /data0_1/MLLM/pope/llava_pope_test.jsonl \
    --image-folder /data0_1/MLLM/pope/val2014 \
    --answers-file /data/MLLM/pope/answers/llava-v1.5-7b_visiondrop.jsonl \
    --temperature 0 \
    --layer_list [8,16,24] \
    --image_token_list [[30,5],[22,4],[16,3]] \
    --dominant 42 \
    --contextual 6 \
    --conv-mode vicuna_v1

python llava/eval_visiondrop2/eval_pope.py \
    --annotation-dir /data0_1/MLLM/pope/coco \
    --question-file /data0_1/MLLM/pope/llava_pope_test.jsonl \
    --result-file /data/MLLM/pope/answers/llava-v1.5-7b_visiondrop.jsonl
