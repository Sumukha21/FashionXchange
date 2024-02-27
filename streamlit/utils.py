import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch
import cv2
import matplotlib.pyplot as plt
import argparse
import groundingdino.datasets.transforms as T
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from torchvision.ops import box_convert


def show_mask(mask, image, random_color=True):
    """
    Overlay a mask on an image and return the composited result.

    Args:
        mask (torch.Tensor): Mask to overlay.
        image (np.ndarray): Image to overlay the mask on.
        random_color (bool, optional): If True, overlay with random color.
                                      If False, use a fixed color. Default is True.

    Returns:
        np.ndarray: Image with the mask overlaid.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def transform_boxes(predictor,boxes, src,device):
    """
    Transform boxes to adjust to the source image dimensions.

    Args:
        boxes (torch.Tensor): Bounding boxes in the format [x_center, y_center, width, height].
        src (np.ndarray): Source image.

    Returns:
        torch.Tensor: Transformed boxes.
    """
    H, W, _ = src.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    return predictor.transform.apply_boxes_torch(boxes_xyxy, src.shape[:2]).to(device)

def save_image(image, file_path):
    """
    Save an image to the specified file path.

    Args:
        image (PIL.Image.Image): Image to be saved.
        file_path (str): Path where the image will be saved.
    """
    try:
        image.save(file_path)
        print(f"Image saved: {file_path}")
    except Exception as e:
        print(f"Error saving image to {file_path}: {e}")

def edit_image(path, item, prompt, box_threshold, text_threshold):
    """
    Edit an image by replacing objects using segmentation and inpainting.

    Args:
        path (str): Path to the image file.
        item (str): Object to be recognized in the image.
        prompt (str): Object to replace the selected object in the image.
        box_threshold (float): Threshold for bounding box predictions.
        text_threshold (float): Threshold for text predictions.

    Returns:
        np.ndarray: Edited image.
    """
    src, img = load_image(path)

    # Predict object bounding boxes, logits, and phrases
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=img,
        caption=item,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # Set up predictor
    sam_predictor.set_image(src)
    new_boxes = transform_boxes(sam_predictor,boxes, src,device)

    # Predict masks and annotations
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=new_boxes,
        multimask_output=False,
    )

    # Overlay mask on annotated image
    img_annotated_mask = show_mask(
        masks[0][0].cpu(),
        annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]
    )

    # Apply inpainting pipeline
    edited_image = pipeline(prompt=prompt,
                        image=Image.fromarray(src).resize((512, 512)),
                        mask_image=Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))
    ).images[0]

    return edited_image

def get_mask(groundingdino_model,sam_predictor, image_path, looking_for, device, box_threshold=0.3, text_threshold=0.25):
  fig, axs = plt.subplots(1, 3, figsize=(12, 4))
  src, img = load_image(image_path)
  axs[0].imshow(src, cmap='gray')
  axs[0].set_title('Source Image')
  axs[0].axis('off')
  boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=img,
        caption=looking_for,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
  annotated_frame = annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)
  axs[1].imshow(np.flip(annotated_frame, 2), cmap='gray')
  axs[1].set_title('DINO o/p')
  axs[1].axis('off')

  sam_predictor.set_image(src)
  new_boxes = transform_boxes(sam_predictor,boxes, src, device)
  masks, _, _ = sam_predictor.predict_torch(
      point_coords=None,
      point_labels=None,
      boxes=new_boxes,
      multimask_output=False,
  )
  masks = torch.any(masks, dim=0, keepdim=True)
  img_annotated_mask = show_mask(
        masks[0][0].cpu(),
        annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]
    )
  axs[2].imshow(img_annotated_mask, cmap='gray')
  axs[2].set_title('SAM o/p')
  axs[2].axis('off')
  return masks, src

def dino_person_prediction(img, groundingdino_model, device):
    boxes_og, logits, phrases = predict(
        model=groundingdino_model,
        image=img,
        caption="person",
        box_threshold=0.3,
        text_threshold=0.25,
        device=device
    )
    return boxes_og, logits, phrases

def find_person_in_image(src, boxes_og):
    h, w, _ = src.shape
    boxes = (boxes_og * torch.Tensor([w, h, w, h])).to(torch.int16)
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    boxes = np.array(boxes, dtype=np.int16)
    # print(person_boxes, shirt)
    if len(boxes) > 1:
        person, selected_id = get_person_choice(src, [boxes[i] for i in boxes])
        box = boxes[selected_id]
        # box_og = boxes_og[person_boxes[selected_id]]
    else:    
        print(boxes[0])
        box = boxes[0]
        # box_og = boxes_og[person_boxes[0]]
        person = src[box[1]: box[3], box[0]: box[2]]
        print("The outfit change will be made to the following person: ")
        plt.imshow(person)
    return person, box


def get_outfit_in_person(groundingdino_model, person, attribute, device):
    person_pil = Image.fromarray(person)
    person_transformed, _ = transform(person_pil, None)
    boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=person_transformed,
            caption=attribute,
            box_threshold=0.3,
            text_threshold=0.25,
            device=device
        )
    return boxes, logits, phrases


def get_person_choice(src, boxes):
    fig, axs = plt.subplots((len(boxes) // 3) + 1, 3, figsize=(20, 20))
    people_list = []
    for i in range(len(boxes)):
        column = ((i + 1) % 3) - 1
        row = i // 3
        box = [int(j) for j in boxes[i]]
        person_i = src[box[1]: box[3], box[0]: box[2], :]
        people_list.append(person_i)
        axs[row, column].imshow(person_i, cmap='gray')
        axs[row, column].set_title(f'person_{i}')
    plt.show()
    print("Multiple people found in the image. Please select which person to chose based on the ID: ")
    selected_index = input()
    return people_list[int(selected_index)], int(selected_index)


def get_outfit_in_person(person, attribute, groundingdino_model, device):
    # person_pil = Image.fromarray(person)
    person_transformed, _ = transform(person, None)
    boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=person_transformed,
            caption=attribute,
            box_threshold=0.3,
            text_threshold=0.25,
            device=device
        )
    return boxes, logits, phrases


def select_required_clothing(ref_box, boxes):
    ref_box = torch.tensor([ref_box])
    boxes = torch.Tensor(boxes)
    x1 = torch.max(ref_box[:, 0], boxes[:, 0])
    y1 = torch.max(ref_box[:, 1], boxes[:, 1])
    x2 = torch.min(ref_box[:, 0] + ref_box[:, 2], boxes[:, 0] + boxes[:, 2])
    y2 = torch.min(ref_box[:, 1] + ref_box[:, 3], boxes[:, 1] + boxes[:, 3])
    width = torch.clamp(x2 - x1, min=0)
    height = torch.clamp(y2 - y1, min=0)
    intersection_areas = width * height
    return int(torch.argmax(intersection_areas))

transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
