#!/bin/bash

MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
INSTANCE_DIR="C:\Users\smanjun3\Desktop\FashionXchange\text2human\images"
INSTANCE_CAPTION_FILE="C:\Users\smanjun3\Desktop\FashionXchange\text2_human_BLIP_dict.json"
INSTANCE_IMAGE_MASKS_DIR="C:\Users\smanjun3\Desktop\FashionXchange\text2human\Masks"
CLASS_DATA_DIR="C:\Users\smanjun3\Desktop\FashionXchange\flickr30k-images"
CLASS_IMAGES_CAPTIONS_FILE="C:\Users\smanjun3\Desktop\FashionXchange\flickr_refined_captions.json"
OUTPUT_DIR="C:\Users\smanjun3\Desktop\FashionXchange\Output\accelerator"
LOGGING_DIR="C:\Users\smanjun3\Desktop\FashionXchange\Logs"
PIPELINE_CHECKPOINTS_OUTPUT_DIR="C:\Users\smanjun3\Desktop\FashionXchange\Output\pipeline"


  
accelerate launch train_dreambooth_inpaint_lora.py \
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
  --train_batch_size=2 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=20 \
  --accelerator_state_checkpointing_steps=51718 \
  --pipeline_checkpoints_output_dir=$PIPELINE_CHECKPOINTS_OUTPUT_DIR  \
  --pipeline_checkpointing_steps=20688 \
  --rank=4
