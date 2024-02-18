#!/bin/bash

MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
INSTANCE_DIR="/content/text2human_images/images"
INSTANCE_CAPTION_FILE="/content/drive/Shareddrives/lora/text2human/text2_human_BLIP_dict.json"
INSTANCE_IMAGE_MASKS_DIR="/content/text2human_masks/masks"
CLASS_DATA_DIR="/content/flickr30k_images/flickr30k-images"
CLASS_IMAGES_CAPTIONS_FILE="/content/drive/Shareddrives/lora/flick30k/flickr_refined_captions.json"
OUTPUT_DIR="/content/outputs/"
LOGGING_DIR="/content/outputs/logs"
PIPELINE_CHECKPOINTS_OUTPUT_DIR="/content/outputs/pipeline"


accelerate launch /content/FashionXchange/src/train_dreambooth_inpaint_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_text_encoder \
  --with_prior_preservation \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_image_captions_file=$INSTANCE_CAPTION_FILE \
  --instance_images_mask_dir=$INSTANCE_IMAGE_MASKS_DIR \
  --class_data_dir=$CLASS_DATA_DIR \
  --class_image_captions_file=$CLASS_IMAGES_CAPTIONS_FILE \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$LOGGING_DIR \
  --resolution=512 \
  --rank=4 \
  --train_batch_size=2 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=20 \
  --pipeline_checkpoints_output_dir=$PIPELINE_CHECKPOINTS_OUTPUT_DIR  \
  --pipeline_checkpointing_steps=20688 \
