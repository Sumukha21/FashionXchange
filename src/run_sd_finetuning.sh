#!/bin/bash

MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
INSTANCE_DIR="C:\Users\smanjun3\Desktop\FashionXchange\text2human\images"
OUTPUT_DIR="C:\Users\smanjun3\Desktop\FashionXchange\Output\accelerator"
LOGGING_DIR="C:\Users\smanjun3\Desktop\FashionXchange\Logs"
INSTANCE_CAPTION_FILE="C:\Users\smanjun3\Desktop\FashionXchange\text2human\masked_image_captions.json" #"C:\Users\smanjun3\Desktop\FashionXchange\2667_IMAGES_dict.json"
PIPELINE_CHECKPOINTS_OUTPUT_DIR="C:\Users\smanjun3\Desktop\FashionXchange\Output\pipeline"
INSTANCE_IMAGE_MASKS_DIR="C:\Users\smanjun3\Desktop\FashionXchange\text2human\Masks"

  
accelerate launch src/sd_inpainting_finetuning.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_image_captions_file=$INSTANCE_CAPTION_FILE \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$LOGGING_DIR \
  --resolution=512 \
  --train_batch_size=4 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=10 \
  --accelerator_state_checkpointing_steps=500 \
  --pipeline_checkpoints_output_dir=$PIPELINE_CHECKPOINTS_OUTPUT_DIR  \
  --pipeline_checkpointing_steps=100 \
  --instance_images_mask_dir=$INSTANCE_IMAGE_MASKS_DIR