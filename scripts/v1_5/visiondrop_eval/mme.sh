scripts/v1_5/ori_eval#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m llava.eval_visiondrop.model_vqa_loader \
    --model-path /data0_1/MLLM/liuhaotian/llava-v1.5-7b \
    --question-file /data0_1/MLLM/MME/llava_mme.jsonl \
    --image-folder /data0_1/MLLM/MME/MME_Benchmark_release_version \
    --answers-file /data/MLLM/MME/answers/llava-v1.5-7b_visiondrop.jsonl \
    --temperature 0 \
    --layer_list [8,16,24] \
    --image_token_list [[30,5],[22,4],[16,3]] \
    --dominant 42 \
    --contextual 6 \
    --conv-mode vicuna_v1


cd /data0_1/MLLM/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b_visiondrop

cd eval_tool

python calculation.py --results_dir /data/MLLM/MME/answers/llava-v1.5-7b_visiondrop

