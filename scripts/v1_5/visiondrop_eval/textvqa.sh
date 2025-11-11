#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m llava.eval_visiondrop.model_vqa_loader \
    --model-path /data0_1/MLLM/liuhaotian/llava-v1.5-7b \
    --question-file /data0_1/MLLM/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /data0_1/MLLM/textvqa/train_images \
    --answers-file /data/MLLM/textvqa/answers/llava-v1.5-7b_visiondrop.jsonl \
    --temperature 0 \
    --layer_list [8,16,24] \
    --image_token_list [[30,5],[22,4],[16,3]] \
    --dominant 42 \
    --contextual 6 \
    --conv-mode vicuna_v1


python -m llava.eval_visiondrop.eval_textvqa \
    --annotation-file /data0_1/MLLM/textvqa/TextVQA_0.5.1_val.json \
    --result-file /data/MLLM/textvqa/answers/llava-v1.5-7b_visiondrop.jsonl

