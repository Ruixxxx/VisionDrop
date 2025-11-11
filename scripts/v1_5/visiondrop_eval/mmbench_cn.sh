#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SPLIT="MMBench_TEST_CN_legacy"

python -m llava.eval_visiondrop.model_vqa_mmbench \
    --model-path /data0_1/MLLM/liuhaotian/llava-v1.5-7b \
    --question-file /data0_1/MLLM/mmbench_cn/$SPLIT.tsv \
    --answers-file /data/MLLM/mmbench_cn/answers/$SPLIT/llava-v1.5-7b_visiondrop.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list [8,16,24] \
    --image_token_list [[30,5],[22,4],[16,3]] \
    --dominant 42 \
    --contextual 6 \
    --conv-mode vicuna_v1

mkdir -p /data/MLLM/mmbench_cn/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /data0_1/MLLM/mmbench_cn/$SPLIT.tsv \
    --result-dir /data/MLLM/mmbench_cn/answers/$SPLIT \
    --upload-dir /data/MLLM/mmbench_cn/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b_visiondrop

