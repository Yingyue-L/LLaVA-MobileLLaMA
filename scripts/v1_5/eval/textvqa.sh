#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-base /data/yingyueli/hub/MobileLLaMA-1.4B-Chat \
    --model-path checkpoints/llava-MobileLLaMA-1.4B-Chat-pretrain \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-MobileLLaMA-1.4B-Chat-pretrain.jsonl \
    --temperature 0 \
    --num_beams 5 \
    --max_new_tokens 10 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-MobileLLaMA-1.4B-Chat-pretrain.jsonl
