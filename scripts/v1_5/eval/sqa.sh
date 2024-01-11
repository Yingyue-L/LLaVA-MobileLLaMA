#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-base /data/yingyueli/hub/MobileLLaMA-1.4B-Chat \
    --model-path checkpoints/llava-MobileLLaMA-1.4B-Chat-lora \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-MobileLLaMA-1.4B-Chat-lora.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-MobileLLaMA-1.4B-Chat-lora.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-MobileLLaMA-1.4B-Chat-lora_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-MobileLLaMA-1.4B-Chat-lora_result.json

ds="scienceqa_test_img"
PYTHONPATH=/data/yingyueli/LLaVA_MobileLLaMA python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-4} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    llava/eval/evaluate_multiple_choice.py \
    --model-base /data/yingyueli/hub/MobileLLaMA-1.4B-Chat \
    --model-path checkpoints/llava-MobileLLaMA-1.4B-Chat-lora \
    --dataset $ds \
    --batch-size 4 \
    --num-workers 2