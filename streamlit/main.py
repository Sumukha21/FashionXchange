import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
import groundingdino.datasets.transforms as T
import argparse
from utils import show_mask, transform_boxes, save_image, edit_image, get_mask, dino_person_prediction

st.title('FashionXchange')

device = torch.device("cuda")

# Paths
sam_checkpoint_path = "/home/kganapa/projects/FashioXchange/weights/sam_vit_h_4b8939.pth"
groundingdino_model_path = "/home/kganapa/projects/FashioXchange/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
groundingdino_weights_path = "/home/kganapa/projects/FashioXchange/weights/groundingdino_swint_ogc.pth"

# SAM Parameters
model_type = "vit_h"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device=device)

@st.cache_resource
def load_sam_model():
    return SamPredictor(sam_model)

sam_predictor = load_sam_model()

# Stable Diffusion
@st.cache_resource
def load_pipeline():
    return StableDiffusionInpaintPipeline.from_pretrained("/home/kganapa/projects/FashioXchange/stable-diffusion-2-inpainting",
                                                     torch_dtype=torch.float16).to(device)

pipeline = load_pipeline()

# Grounding DINO
@st.cache_resource
def load_gdino():
    return load_model(groundingdino_model_path, groundingdino_weights_path)

groundingdino_model = load_gdino()

image_path = "/home/kganapa/projects/FashioXchange/MEN-Denim-id_00000089-01_7_additional.jpg"
# looking_for = "clothes, arms"
# masks, src = get_mask(groundingdino_model,sam_predictor, image_path, looking_for, device,  box_threshold=0.3, text_threshold=0.25)


with st.form("Input"):
    with st.sidebar:
        st.write("Upload file")
        uploaded_file = st.file_uploader("Choose a file")
        looking_for= st.selectbox('Select dress areas to modify:', ('Clothes, arms, legs', 'Shirt, arms', 'Pant, legs'))
        input_string = st.text_input("Enter prompt to make changes:", "A man wearing a blue shirt")
        submitted = st.form_submit_button("Submit")
    if submitted:
        with st.spinner('Processing Image...'):
            # change_required = input_string
            # masks, src = get_mask(groundingdino_model,sam_predictor, uploaded_file, 
            #                       looking_for, device,  box_threshold=0.3, text_threshold=0.25)
            # edited_image = pipeline(prompt=change_required,
            #                         image=Image.fromarray(src).resize((512, 512)),
            #                         mask_image=Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))
            #     ).images[0]
            src, boxes_og, logits, phrases = dino_person_prediction(uploaded_file, groundingdino_model, device)
            all_preds = annotate(src, boxes_og, logits, phrases)
            st.image(all_preds)
            
        # col1, col2 = st.columns(2)
        # # Display first image and its heading
        # with col1:
        #     st.header("Original Image")
        #     st.image(image_path)
            

        # # Display second image and its heading
        # with col2:
        #     st.header("Edited Image")
        #     st.image(edited_image)
            