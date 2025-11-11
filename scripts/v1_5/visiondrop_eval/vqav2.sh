#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval_visiondrop.model_vqa_loader \
        --model-path /data0_1/MLLM/liuhaotian/llava-v1.5-7b \
        --question-file /data0_1/MLLM/vqav2/$SPLIT.jsonl \
        --image-folder /data0_1/MLLM/vqav2/test2015 \
        --answers-file /data/MLLM/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --layer_list [8,16,24] \
        --image_token_list [[30,5],[22,4],[16,3]] \
        --dominant 42 \
        --contextual 6 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/data/MLLM/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /data/MLLM/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT


