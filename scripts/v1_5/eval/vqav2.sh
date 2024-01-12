#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-MobileLLaMA-1.4B-Chat-lora-665k-retry"
SPLIT="llava_vqav2_val"
DATASET="vqav2_val"
CONV="vicuna_v1"

# IDX=0
# CHUNKS=2
# python -m llava.eval.model_vqa_loader \
#     --model-base /data/yingyueli/hub/MobileLLaMA-1.4B-Chat \
#     --model-path checkpoints/llava-MobileLLaMA-1.4B-Chat-lora-665k-retry \
#     --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#     --image-folder ./playground/data/eval/vqav2/val2014 \
#     --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_${CONV}_short.jsonl \
#     --num-chunks 1 \
#     --chunk-idx 0 \
#     --temperature 0 \
#     --conv-mode vicuna_v1


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-base /data/yingyueli/hub/MobileLLaMA-1.4B-Chat \
        --model-path checkpoints/llava-MobileLLaMA-1.4B-Chat-lora-665k-retry \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vqav2/val2014 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_${CONV}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 3 \
        --max_new_tokens 10 \
        --conv-mode $CONV &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}_${CONV}.jsonl >> "$output_file"
done

CKPT="llava-MobileLLaMA-1.4B-Chat-lora-665k-retry"
SPLIT="llava_vqav2_val"
DATASET="vqav2_val"
CONV="vicuna_v1"

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

python llava/eval/evaluate_vqa.py --dataset $DATASET --results-file ./playground/data/eval/vqav2/answers_upload/$SPLIT/$CKPT.json