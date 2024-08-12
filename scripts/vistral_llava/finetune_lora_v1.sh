#!/bin/bash

# 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 

# Version 1: baseline
source /home/lhbac/anaconda3/bin/activate
conda activate llava 

python /home/lhbac/vic/Vistral-V/scripts/vistral_llava/login.py

export WANDB_API_KEY=3317e3c4d2ff3c0067ec4fca33075e1c056116e4

deepspeed --include localhost:3,4 /home/lhbac/vic/Vistral-V/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0.00000125 \
    --deepspeed /home/lhbac/vic/Vistral-V/scripts/zero3.json \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --version vistral \
    --data_path /home/lhbac/vic/Vistral-V/scripts/vistral_llava/data/vi_llava_train.json \
    --image_folder /home/lhbac/vic/Vistral-V/scripts/vistral_llava/data/images \
    --vision_tower google/siglip-base-patch16-256-multilingual \
    --pretrain_mm_mlp_adapter /home/lhbac/vic/Vistral-V/scripts/vistral_llava/.checkpoints/llava-vistral-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/lhbac/vic/Vistral-V/scripts/vistral_llava/.checkpoints/llava-vistral-lora-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 0.0000125 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb