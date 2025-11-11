#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m llava.eval_visiondrop.model_vqa_loader \
    --model-path /data0_1/MLLM/liuhaotian/llava-v1.5-7b \
    --question-file /data0_1/MLLM/vizwiz/llava_test.jsonl \
    --image-folder /data0_1/MLLM/vizwiz/test \
    --answers-file /data/MLLM/vizwiz/answers/llava-v1.5-7b_visiondrop.jsonl \
    --temperature 0 \
    --layer_list [8,16,24] \
    --image_token_list [[30,5],[22,4],[16,3]] \
    --dominant 42 \
    --contextual 6 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /data0_1/MLLM/vizwiz/llava_test.jsonl \
    --result-file /data/MLLM/vizwiz/answers/llava-v1.5-7b_visiondrop.jsonl \
    --result-upload-file /data/MLLM/vizwiz/answers_upload/llava-v1.5-7b_visiondrop.json
