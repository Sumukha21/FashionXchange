"""
    Source: https://github.com/huggingface/diffusers/blob/main/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py
    Modifications:
        1. DreamBoothDataset replaced with TrainDataset: 
            Allows loading images with corresponding prompts for both instance and class images
            (DreamBoothDataset had a single prompt for class images and another prompt for instance images)
        2. PromptDataset replaced with ClassDatasetSampleGeneraion:
            Allows loading of prompts using which class samples can be generated for prior preservation loss
"""


import os
import math
import torch
import itertools
import random
import json
from glob import glob
import argparse
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pathlib import Path
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from huggingface_hub.utils import insecure_hashlib
from huggingface_hub import create_repo, upload_folder
from diffusers.optimization import get_scheduler
# from src.utils.utils import text_file_reader
# from src.datasets import TargetedMaskingDataset


logger = get_logger(__name__)


def text_file_reader(file_path):
    """
    Read the contents of a text file.
    Args:
        file_path (str): The path to the text file.
    Returns:
        list: A list containing the lines of text from the file.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        required=False, # True
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\images",
        required=False, #True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=28, #100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--accelerator_state_checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--class_images_generation_prompts_file",
        type=str,
        default=None,
        help=(
            """File containig=ng the prompts for generating class images using pre-trained Stable 
            Diffusion Inpaintnig model for prior preservation loss"""
        ),
    )
    parser.add_argument(
        "--instance_image_captions_file",
        type=str,
        default=r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\masked_image_captions.json",
        required=False, #True,
        help=(
            "Path to json/yaml/text file containig the text prompts for the instance images"
        )    
    )
    parser.add_argument(
        "--class_image_captions_file",
        type=str,
        default=None,
        help=(
            "Path to json/yaml/text file containig the text prompts for the class images"
        )
    )

    parser.add_argument(
        "--pipeline_checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "The frequency in terms of number of training steps at which the pipeline weights need to be stored"
        )
    )

    parser.add_argument(
        "--pipeline_checkpoints_output_dir",
        type=str,
        default=None,
        help="The output directory where the pipeline weights will be stored",
    )

    parser.add_argument(
        "--instance_images_mask_dir",
        type=str,
        default=r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\Masks",
        help="The directory where the instance image masks are stored",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        # if args.class_prompt is None:
        #     raise ValueError("You must specify prompt for class images.")

    return args


class ClassDatasetSampleGeneraion(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompts_for_generation_path):
        self.prompt_list = text_file_reader(prompts_for_generation_path)

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt_list[index]
        example["index"] = index
        return example


class TrainDataset(Dataset):
    def __init__(
            self, 
            instance_image_captions_file, 
            instance_image_dir, 
            tokenizer,
            image_size=512, 
            center_crop=False, 
            class_image_dir=None, 
            class_image_prompts_file=None, 
            class_sample_generated_prompts=None):
        
        self.tokenizer = tokenizer
        self.instance_image_captions = self.caption_file_reader(instance_image_captions_file)
        self._length = len(self.instance_image_captions.keys())
        # self.instance_image_list = glob(os.path.join(instance_image_dir, '*.jpg'))
        self.instance_image_list = [os.path.join(instance_image_dir, image_file) for image_file in self.instance_image_captions.keys()]

        if class_image_prompts_file is not None:
            self.class_image_captions = self.caption_file_reader(class_image_prompts_file)
        else:
            self.class_image_captions = dict()

        if class_sample_generated_prompts is not None:
            self.class_image_captions.update(class_sample_generated_prompts)
       

        if class_image_dir is not None:
            self.class_image_list = glob(os.path.join(class_image_dir, '*.jpeg'))
        else:
            self.class_image_list = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length
    
    @staticmethod
    def caption_file_reader(captions_file_path, image_directory=None):
        ext = os.path.splitext(captions_file_path)[1]
        if captions_file_path.endswith(".txt"):
            assert image_directory is not None
            captions = text_file_reader(captions_file_path)
            image_vs_captions = dict()
            img_list = glob(os.path.join(image_directory, '*.jpg'))
            for img_path, caption in zip(img_list, captions):
                img_name = os.path.basename(img_path)
                image_vs_captions[img_name] = caption
            return image_vs_captions
        elif captions_file_path.endswith(".json"):
            """
            Code for creating a dictionary of image_name vs captions by reading a json file 
            """
            with open(captions_file_path, "r") as json_file:
                image_vs_captions = json.load(json_file)
            return image_vs_captions

        elif captions_file_path.endswith(".yaml"):
            """
            Code for creating a dictionary of image_name vs captions by reading a yaml file 
            """
            pass

    def __getitem__(self, idx):
        example = dict()
        image_name = os.path.basename(self.instance_image_list[idx])
        instance_image = Image.open(self.instance_image_list[idx])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)
        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_image_captions[image_name],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        if self.class_image_list is not None:
            class_image_name = os.path.basename(self.class_image_list[idx])
            class_image = Image.open(self.class_image_list[idx])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms_resize_and_crop(class_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(
                self.class_image_captions[class_image_name],
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        return example


class TargetedMaskingDataset(Dataset):
    def __init__(self, 
                instance_image_captions_file, 
                instance_image_dir,
                instance_images_mask_dir, 
                tokenizer,
                image_size=512):
        self.tokenizer = tokenizer
        self.instance_image_captions = self.caption_file_reader(instance_image_captions_file)
        self.mask_directory = instance_images_mask_dir
        self.instance_image_list = [os.path.join(instance_image_dir, image_file) for image_file in self.instance_image_captions.keys()]
        self.image_transforms_resize = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(image_size)
            ]
        )
        self.image_transforms_resize2 = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                # transforms.CenterCrop(image_size)
            ]
        )
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.instance_image_captions)

    @staticmethod
    def caption_file_reader(captions_file_path, image_directory=None):
        ext = os.path.splitext(captions_file_path)[1]
        if captions_file_path.endswith(".txt"):
            assert image_directory is not None
            captions = text_file_reader(captions_file_path)
            image_vs_captions = dict()
            img_list = glob(os.path.join(image_directory, '*.jpg'))
            for img_path, caption in zip(img_list, captions):
                img_name = os.path.basename(img_path)
                image_vs_captions[img_name] = caption
            return image_vs_captions
        elif captions_file_path.endswith(".json"):
            """
            Code for creating a dictionary of image_name vs captions by reading a json file 
            """
            with open(captions_file_path, "r") as json_file:
                image_vs_captions = json.load(json_file)
            return image_vs_captions

        elif captions_file_path.endswith(".yaml"):
            """
            Code for creating a dictionary of image_name vs captions by reading a yaml file 
            """
            pass
    
    def __getitem__(self, idx):
        example = dict()
        image_name = os.path.basename(self.instance_image_list[idx])
        instance_image = Image.open(self.instance_image_list[idx])
        mask_paths = os.listdir(os.path.join(self.mask_directory, image_name.split(".")[0]))
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        random_selector = np.random.randint(0, len(mask_paths))
        # print(mask_paths[random_selector])
        selected_mask = Image.open(os.path.join(self.mask_directory, os.path.join(image_name.split(".jpg")[0], mask_paths[random_selector])))
        example["mask"] = self.image_transforms_resize2(selected_mask)
        instance_image = self.image_transforms_resize(instance_image)
        example["PIL_image"] = instance_image
        example["instance_image"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_image_captions[image_name],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example


class TargetedMaskingDatasetWithPriorPreservation(Dataset):
    def __init__(self, 
                instance_image_captions_file, 
                instance_image_dir,
                instance_images_mask_dir, 
                class_images_dir,
                class_image_captions_file,
                tokenizer,
                image_size=512):
        self.tokenizer = tokenizer
        self.instance_image_captions = self.caption_file_reader(instance_image_captions_file)
        self.mask_directory = instance_images_mask_dir
        self.instance_images_mask_list = os.listdir(self.mask_directory)
        self.instance_image_captions_updated = dict()
        for i in self.instance_images_mask_list:
            self.instance_image_captions_updated[i] = self.instance_image_captions[i]
        self.instance_image_captions = self.instance_image_captions_updated
        self.instance_image_list = [os.path.join(instance_image_dir, image_file + '.jpg') for image_file in self.instance_image_captions.keys()]
        self.class_image_captions = self.caption_file_reader(class_image_captions_file)
        self.class_image_list = [os.path.join(class_images_dir, image_file) for image_file in self.class_image_captions.keys()]
        if len(self.class_image_list) > len(self.instance_image_list):
            self.class_image_list = random.sample(self.class_image_list, len(self.instance_image_list))
        elif len(self.class_image_list) < len(self.instance_image_list):
            required_samples_count = len(self.instance_image_list) - len(self.class_image_list)
            while required_samples_count > 0:
                if required_samples_count > len(self.class_image_list):
                    class_image_list = random.sample(self.class_image_list, len(self.class_image_list))
                    required_samples_count -= (len(self.class_image_list))
                else:
                    class_image_list = random.sample(self.class_image_list, required_samples_count)
                    required_samples_count = 0
                self.class_image_list.extend(class_image_list)
    
        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(image_size),
            ]
        )
        
        self.image_transforms_resize = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(image_size)
            ]
        )
        self.image_transforms_resize2 = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                # transforms.CenterCrop(image_size)
            ]
        )
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.instance_image_captions)

    @staticmethod
    def caption_file_reader(captions_file_path, image_directory=None):
        ext = os.path.splitext(captions_file_path)[1]
        if captions_file_path.endswith(".txt"):
            assert image_directory is not None
            captions = text_file_reader(captions_file_path)
            image_vs_captions = dict()
            img_list = glob(os.path.join(image_directory, '*.jpg'))
            for img_path, caption in zip(img_list, captions):
                img_name = os.path.basename(img_path)
                image_vs_captions[img_name] = caption
            return image_vs_captions
        elif captions_file_path.endswith(".json"):
            """
            Code for creating a dictionary of image_name vs captions by reading a json file 
            """
            with open(captions_file_path, "r") as json_file:
                image_vs_captions = json.load(json_file)
            return image_vs_captions

        elif captions_file_path.endswith(".yaml"):
            """
            Code for creating a dictionary of image_name vs captions by reading a yaml file 
            """
            pass
        elif captions_file_path.endswith(".token"):
            with open(captions_file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
            image_vs_caption = dict()
            for line in content:
                image_name, caption = line.split("\n")[0].split("\t")
                image_name = image_name.split("#")[0]
                if image_vs_caption.get(image_name):
                    continue
                image_vs_caption[image_name] = caption
            return image_vs_caption
    
    def __getitem__(self, idx):
        example = dict()
        image_name = os.path.basename(self.instance_image_list[idx])
        instance_image = Image.open(self.instance_image_list[idx])
        mask_paths = os.listdir(os.path.join(self.mask_directory, image_name.split(".")[0]))
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        random_selector = np.random.randint(0, len(mask_paths))
        # print(mask_paths[random_selector])
        selected_mask = Image.open(os.path.join(self.mask_directory, os.path.join(image_name.split(".jpg")[0], mask_paths[random_selector])))
        example["mask"] = self.image_transforms_resize2(selected_mask)
        instance_image = self.image_transforms_resize(instance_image)
        example["PIL_image"] = instance_image
        example["instance_image"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_image_captions[image_name.strip('.jpg')],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        class_image_name = os.path.basename(self.class_image_list[idx])
        class_image = Image.open(self.class_image_list[idx])
        if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
        class_image = self.image_transforms_resize_and_crop(class_image)
        example["class_images"] = self.image_transforms(class_image)
        example["class_PIL_images"] = class_image
        example["class_prompt_ids"] = self.tokenizer(
            self.class_image_captions[class_image_name],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example
          

def random_perturb_mask(mask):
    ones = np.where(mask > 0)
    mask[mask > 0] = 1
    y_max, y_min, x_max, x_min = max(ones[0]), min(ones[0]), max(ones[1]), min(ones[1])
    increase_or_decrease = [0, 1]
    up_down_left_right = [0, 1, 2, 3]
    choice_increase_or_decrease = np.random.choice(increase_or_decrease)
    choice_udlr = np.random.choice(up_down_left_right)
    if choice_udlr == 0 or choice_udlr == 1:
        increase_or_decrease_quantity = np.random.randint(50, 100)
    elif choice_udlr == 2 or choice_udlr == 3:
        increase_or_decrease_quantity = np.random.randint(50, 100)
    # print(choice_increase_or_decrease, choice_udlr, increase_or_decrease_quantity)
    if choice_increase_or_decrease == 0:
        if choice_udlr == 0:
            increase_or_decrease_quantity = min(increase_or_decrease_quantity, y_min)
            mask[y_min - increase_or_decrease_quantity: y_min + (y_max - y_min) // 6, x_min: x_max + 1] = True
        elif choice_udlr == 1:
            increase_or_decrease_quantity = min(increase_or_decrease_quantity, mask.shape[0] - y_max)
            mask[y_max - (y_max - y_min) // 6: y_max + increase_or_decrease_quantity, x_min: x_max + 1] = True
        elif choice_udlr == 2:
            increase_or_decrease_quantity = min(x_min, increase_or_decrease_quantity)
            mask[y_min: y_max + 1, x_min - increase_or_decrease_quantity: x_min + (x_max - x_min) // 4] = True
        else:
            increase_or_decrease_quantity = min(mask.shape[1] - x_max, increase_or_decrease_quantity)
            mask[y_min: y_max + 1, x_max - (x_max - x_min) // 4: x_max + increase_or_decrease_quantity] = True
    elif choice_increase_or_decrease == 1:
        if choice_udlr == 0:
            mask[y_min: y_min + increase_or_decrease_quantity, x_min: x_max + 1] = False
        elif choice_udlr == 1:
            mask[y_max: y_max - increase_or_decrease_quantity, x_min: x_max + 1] = False
        elif choice_udlr == 2:
            mask[y_min: y_max + 1, x_min: x_min + increase_or_decrease_quantity] = False
        else:
            mask[y_min: y_max + 1, x_max - increase_or_decrease_quantity: x_max] = False
    return mask


def model_finetuning():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    args.train_text_encoder = True

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation and args.class_images_generation_prompts_file:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
        )
        pipeline.set_progress_bar_config(disable=True)
        sample_dataset = ClassDatasetSampleGeneraion(args.class_images_generation_prompts_file)
        logger.info(f"Number of class images to sample: {len(sample_dataset)}.")
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset, batch_size=args.sample_batch_size, num_workers=1
        )

        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(accelerator.device)
        transform_to_pil = transforms.ToPILImage()
        for j, example in tqdm(
            enumerate(sample_dataloader), desc="Generating class images", disable=not accelerator.is_local_main_process
        ):
            bsz = len(example["prompt"])
            fake_images = torch.rand((3, args.resolution, args.resolution))
            transform_to_pil = transforms.ToPILImage()
            fake_pil_images = transform_to_pil(fake_images)

            fake_mask = random_mask((args.resolution, args.resolution), ratio=1, mask_full_image=True)

            images = pipeline(prompt=example["prompt"], mask_image=fake_mask, image=fake_pil_images).images
            sample_img_names_prompts = dict()
            for i, image in enumerate(images):
                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = args.class_images_dir / f"{example['index'][i]}-{hash_image}.jpg"
                sample_img_names_prompts[f"{example['index'][i]}-{hash_image}.jpg"] = example["prompt"][i]
                image.save(image_filename)
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    else:
        sample_img_names_prompts = dict()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.instance_images_mask_dir is None:
        train_dataset = TrainDataset(instance_image_captions_file=args.instance_image_captions_file,
                                    instance_image_dir=args.instance_data_dir,
                                    image_size=args.resolution,
                                    tokenizer=tokenizer,
                                    center_crop=args.center_crop,
                                    class_image_dir=args.class_data_dir,
                                    class_image_prompts_file=args.class_image_captions_file,
                                    class_sample_generated_prompts=sample_img_names_prompts)

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # Concat class and instance examples for prior preservation.
            # We do this to avoid doing two forward passes.
            if args.with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]
                pior_pil = [example["class_PIL_images"] for example in examples]

            masks = []
            masked_images = []
            for example in examples:
                pil_image = example["PIL_images"]
                # generate a random mask
                mask = random_mask(pil_image.size, 1, False)
                # prepare mask and masked image
                mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

                masks.append(mask)
                masked_images.append(masked_image)

            if args.with_prior_preservation:
                for pil_image in pior_pil:
                    # generate a random mask
                    mask = random_mask(pil_image.size, 1, False)
                    # prepare mask and masked image
                    mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

                    masks.append(mask)
                    masked_images.append(masked_image)

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
            masks = torch.stack(masks)
            masked_images = torch.stack(masked_images)
            batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
            return batch
    
    elif args.instance_images_mask_dir is not None and args.with_prior_preservation:
        print("Inside Traget Masking Dataset with prior loss")
        train_dataset = TargetedMaskingDatasetWithPriorPreservation(
                                                                    instance_image_captions_file=args.instance_image_captions_file, 
                                                                    instance_image_dir=args.instance_data_dir,
                                                                    instance_images_mask_dir=args.instance_images_mask_dir, 
                                                                    class_images_dir=args.class_data_dir,
                                                                    class_image_captions_file=args.class_image_captions_file,
                                                                    tokenizer=tokenizer,
                                                                    image_size=512
                                                                    )
        
        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_image"] for example in examples]
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            pior_pil = [example["class_PIL_images"] for example in examples]
            masks = []
            masked_images = []
            for example in examples:
                pil_image = np.array(example["PIL_image"])
                mask = np.array(example["mask"])
                mask = random_perturb_mask(mask)
                image = torch.from_numpy(pil_image).to(dtype=torch.float32) / 127.5 - 1.0
                mask = torch.from_numpy(mask)
                mask = mask.unsqueeze(2)
                masked_image = image * (mask.expand(-1, -1, 3) == 0)
                masked_image = masked_image.permute(2, 0, 1)
                mask = mask.permute(2, 0, 1)
                masks.append(mask)
                masked_images.append(masked_image)
            
            for pil_image in pior_pil:
                # generate a random mask
                mask = random_mask(pil_image.size, 1, False)
                # prepare mask and masked image
                mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

                masks.append(mask[0])
                masked_images.append(masked_image[0])

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
            masks = torch.stack(masks)
            masked_images = torch.stack(masked_images)
            batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
            return batch    

    else:
        train_dataset = TargetedMaskingDataset(instance_image_captions_file=args.instance_image_captions_file, 
                                               instance_image_dir=args.instance_data_dir,
                                               instance_images_mask_dir=args.instance_images_mask_dir, 
                                               tokenizer=tokenizer,
                                               image_size=args.resolution)

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_image"] for example in examples]
            masks = []
            masked_images = []
            for example in examples:
                pil_image = np.array(example["PIL_image"])
                mask = np.array(example["mask"])
                mask = random_perturb_mask(mask)
                image = torch.from_numpy(pil_image).to(dtype=torch.float32) / 127.5 - 1.0
                mask = torch.from_numpy(mask)
                mask = mask.unsqueeze(2)
                masked_image = image * (mask.expand(-1, -1, 3) == 0)
                masked_image = masked_image.permute(2, 0, 1)
                mask = mask.permute(2, 0, 1)
                masks.append(mask)
                masked_images.append(masked_image)
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
            masks = torch.stack(masks)
            masked_images = torch.stack(masked_images)
            batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
            return batch
    
    print(f"Training dataset of {len(train_dataset)} images loaded")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space

                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Convert masked images to latent space
                masked_latents = vae.encode(
                    batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                masks = batch["masks"]
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(mask.unsqueeze(0), size=(args.resolution // 8, args.resolution // 8))
                        for mask in masks
                    ]
                )
                mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.accelerator_state_checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if accelerator.is_main_process and global_step % args.pipeline_checkpointing_steps == 0:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                )
                save_folder = os.path.join(os.path.join(args.pipeline_checkpoints_output_dir, f"checkpoint-{global_step}"))
                os.makedirs(save_folder)
                pipeline.save_pretrained(save_folder)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    model_finetuning()
