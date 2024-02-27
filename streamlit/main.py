import streamlit as st
import pandas as pd
import numpy as np
import time
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

device = torch.device("cpu")

# Paths
sam_checkpoint_path = r"C:\Users\sumuk\OneDrive\Desktop\GitHub\FashionXchange\GroundingDINO\weights\sam_vit_h_4b8939.pth"
groundingdino_model_path = r"C:\Users\sumuk\OneDrive\Desktop\GitHub\FashionXchange\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
groundingdino_weights_path = r"C:\Users\sumuk\OneDrive\Desktop\GitHub\FashionXchange\GroundingDINO\weights\groundingdino_swint_ogc.pth"

# SAM Parameters
model_type = "vit_h"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device=device)

@st.cache_resource
def load_sam_model():
    return SamPredictor(sam_model)

# sam_predictor = load_sam_model()

# Stable Diffusion
@st.cache_resource
def load_pipeline():
    return StableDiffusionInpaintPipeline.from_pretrained(r"C:\Users\sumuk\OneDrive\Desktop\GitHub\FashionXchange\checkpoint-82752",
                                                     torch_dtype=torch.float16).to(device)

# pipeline = load_pipeline()

# Grounding DINO
@st.cache_resource
def load_gdino():
    return load_model(groundingdino_model_path, groundingdino_weights_path)

# groundingdino_model = load_gdino()

st.session_state["Data submitted"] = False

with st.form("Input"):
    with st.sidebar:
        st.write("Upload file")
        uploaded_file = st.file_uploader("Choose a file")
        looking_for= st.selectbox('Select dress areas to modify:', ('Clothes, arms, legs', 'Shirt, arms', 'Pant, legs'))
        input_string = st.text_input("Enter prompt to make changes:", "A man wearing a blue shirt")
        submitted = st.form_submit_button("Submit")

    if submitted:
        src, img = load_image(uploaded_file)
        imageLocation = st.empty()
        imageLocation.image(src)        
        # with st.spinner('Processing Image...'):
        #     boxes_og, logits, phrases = dino_person_prediction(img, groundingdino_model, device)
        #     all_preds = annotate(src, boxes_og, logits, phrases)
        #     imageLocation.image(np.flip(all_preds, 2))
        # time.sleep(5)
        # imageLocation.empty()
        st.session_state["Data submitted"] = True
    
    if st.session_state["Data submitted"]:
        selected_person = st.slider('Select person ID', 0, 5, -1)
        while True:
            if selected_person != -1:
                st.write("selected person: ", selected_person)
                break