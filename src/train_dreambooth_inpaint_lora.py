import argparse
import math
import os
import random
from pathlib import Path
import enum
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
# from diffusers.loaders import AttnProcsLayers
# from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.loaders import LoraLoaderMixin
from typing import Any, Dict, Iterable, List, Optional, Union

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

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

class StateDictType(enum.Enum):
    """
    The mode to use when converting state dicts.
    """

    DIFFUSERS_OLD = "diffusers_old"
    KOHYA_SS = "kohya_ss"
    PEFT = "peft"
    DIFFUSERS = "diffusers"


# We need to define a proper mapping for Unet since it uses different output keys than text encoder
# e.g. to_q_lora -> q_proj / to_q
UNET_TO_DIFFUSERS = {
    ".to_out_lora.up": ".to_out.0.lora_B",
    ".to_out_lora.down": ".to_out.0.lora_A",
    ".to_q_lora.down": ".to_q.lora_A",
    ".to_q_lora.up": ".to_q.lora_B",
    ".to_k_lora.down": ".to_k.lora_A",
    ".to_k_lora.up": ".to_k.lora_B",
    ".to_v_lora.down": ".to_v.lora_A",
    ".to_v_lora.up": ".to_v.lora_B",
    ".lora.up": ".lora_B",
    ".lora.down": ".lora_A",
}


DIFFUSERS_TO_PEFT = {
    ".q_proj.lora_linear_layer.up": ".q_proj.lora_B",
    ".q_proj.lora_linear_layer.down": ".q_proj.lora_A",
    ".k_proj.lora_linear_layer.up": ".k_proj.lora_B",
    ".k_proj.lora_linear_layer.down": ".k_proj.lora_A",
    ".v_proj.lora_linear_layer.up": ".v_proj.lora_B",
    ".v_proj.lora_linear_layer.down": ".v_proj.lora_A",
    ".out_proj.lora_linear_layer.up": ".out_proj.lora_B",
    ".out_proj.lora_linear_layer.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

DIFFUSERS_OLD_TO_PEFT = {
    ".to_q_lora.up": ".q_proj.lora_B",
    ".to_q_lora.down": ".q_proj.lora_A",
    ".to_k_lora.up": ".k_proj.lora_B",
    ".to_k_lora.down": ".k_proj.lora_A",
    ".to_v_lora.up": ".v_proj.lora_B",
    ".to_v_lora.down": ".v_proj.lora_A",
    ".to_out_lora.up": ".out_proj.lora_B",
    ".to_out_lora.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

PEFT_TO_DIFFUSERS = {
    ".q_proj.lora_B": ".q_proj.lora_linear_layer.up",
    ".q_proj.lora_A": ".q_proj.lora_linear_layer.down",
    ".k_proj.lora_B": ".k_proj.lora_linear_layer.up",
    ".k_proj.lora_A": ".k_proj.lora_linear_layer.down",
    ".v_proj.lora_B": ".v_proj.lora_linear_layer.up",
    ".v_proj.lora_A": ".v_proj.lora_linear_layer.down",
    ".out_proj.lora_B": ".out_proj.lora_linear_layer.up",
    ".out_proj.lora_A": ".out_proj.lora_linear_layer.down",
    "to_k.lora_A": "to_k.lora.down",
    "to_k.lora_B": "to_k.lora.up",
    "to_q.lora_A": "to_q.lora.down",
    "to_q.lora_B": "to_q.lora.up",
    "to_v.lora_A": "to_v.lora.down",
    "to_v.lora_B": "to_v.lora.up",
    "to_out.0.lora_A": "to_out.0.lora.down",
    "to_out.0.lora_B": "to_out.0.lora.up",
}

DIFFUSERS_OLD_TO_DIFFUSERS = {
    ".to_q_lora.up": ".q_proj.lora_linear_layer.up",
    ".to_q_lora.down": ".q_proj.lora_linear_layer.down",
    ".to_k_lora.up": ".k_proj.lora_linear_layer.up",
    ".to_k_lora.down": ".k_proj.lora_linear_layer.down",
    ".to_v_lora.up": ".v_proj.lora_linear_layer.up",
    ".to_v_lora.down": ".v_proj.lora_linear_layer.down",
    ".to_out_lora.up": ".out_proj.lora_linear_layer.up",
    ".to_out_lora.down": ".out_proj.lora_linear_layer.down",
}

PEFT_TO_KOHYA_SS = {
    "lora_A": "lora_down",
    "lora_B": "lora_up",
    # This is not a comprehensive dict as kohya format requires replacing `.` with `_` in keys,
    # adding prefixes and adding alpha values
    # Check `convert_state_dict_to_kohya` for more
}

PEFT_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_PEFT,
    StateDictType.DIFFUSERS: DIFFUSERS_TO_PEFT,
}

DIFFUSERS_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_DIFFUSERS,
    StateDictType.PEFT: PEFT_TO_DIFFUSERS,
}

KOHYA_STATE_DICT_MAPPINGS = {StateDictType.PEFT: PEFT_TO_KOHYA_SS}

KEYS_TO_ALWAYS_REPLACE = {
    ".processor.": ".",
}

def convert_state_dict(state_dict, mapping):
    r"""
    Simply iterates over the state dict and replaces the patterns in `mapping` with the corresponding values.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        mapping (`dict[str, str]`):
            The mapping to use for conversion, the mapping should be a dictionary with the following structure:
                - key: the pattern to replace
                - value: the pattern to replace with

    Returns:
        converted_state_dict (`dict`)
            The converted state dict.
    """
    converted_state_dict = {}
    for k, v in state_dict.items():
        # First, filter out the keys that we always want to replace
        for pattern in KEYS_TO_ALWAYS_REPLACE.keys():
            if pattern in k:
                new_pattern = KEYS_TO_ALWAYS_REPLACE[pattern]
                k = k.replace(pattern, new_pattern)

        for pattern in mapping.keys():
            if pattern in k:
                new_pattern = mapping[pattern]
                k = k.replace(pattern, new_pattern)
                break
        converted_state_dict[k] = v
    return converted_state_dict



def convert_unet_state_dict_to_peft(state_dict):
    r"""
    Converts a state dict from UNet format to diffusers format - i.e. by removing some keys
    """
    mapping = UNET_TO_DIFFUSERS
    return convert_state_dict(state_dict, mapping)


def convert_state_dict_to_peft(state_dict, original_type=None, **kwargs):
    r"""
    Converts a state dict to the PEFT format The state dict can be from previous diffusers format (`OLD_DIFFUSERS`), or
    new diffusers format (`DIFFUSERS`). The method only supports the conversion from diffusers old/new to PEFT for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
    """
    if original_type is None:
        # Old diffusers to PEFT
        if any("to_out_lora" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS_OLD
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS
        else:
            raise ValueError("Could not automatically infer state dict type")

    if original_type not in PEFT_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    mapping = PEFT_STATE_DICT_MAPPINGS[original_type]
    return convert_state_dict(state_dict, mapping)


def convert_state_dict_to_diffusers(state_dict, original_type=None, **kwargs):
    r"""
    Converts a state dict to new diffusers format. The state dict can be from previous diffusers format
    (`OLD_DIFFUSERS`), or PEFT format (`PEFT`) or new diffusers format (`DIFFUSERS`). In the last case the method will
    return the state dict as is.

    The method only supports the conversion from diffusers old, PEFT to diffusers new for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
        kwargs (`dict`, *args*):
            Additional arguments to pass to the method.

            - **adapter_name**: For example, in case of PEFT, some keys will be pre-pended
                with the adapter name, therefore needs a special handling. By default PEFT also takes care of that in
                `get_peft_model_state_dict` method:
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                but we add it here in case we don't want to rely on that method.
    """
    peft_adapter_name = kwargs.pop("adapter_name", None)
    if peft_adapter_name is not None:
        peft_adapter_name = "." + peft_adapter_name
    else:
        peft_adapter_name = ""

    if original_type is None:
        # Old diffusers to PEFT
        if any("to_out_lora" in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS_OLD
        elif any(f".lora_A{peft_adapter_name}.weight" in k for k in state_dict.keys()):
            original_type = StateDictType.PEFT
        elif any("lora_linear_layer" in k for k in state_dict.keys()):
            # nothing to do
            return state_dict
        else:
            raise ValueError("Could not automatically infer state dict type")

    if original_type not in DIFFUSERS_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f"Original type {original_type} is not supported")

    mapping = DIFFUSERS_STATE_DICT_MAPPINGS[original_type]
    return convert_state_dict(state_dict, mapping)


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)


def _set_state_dict_into_text_encoder(
    lora_state_dict: Dict[str, torch.Tensor], prefix: str, text_encoder: torch.nn.Module
):
    """
    Sets the `lora_state_dict` into `text_encoder` coming from `transformers`.

    Args:
        lora_state_dict: The state dictionary to be set.
        prefix: String identifier to retrieve the portion of the state dict that belongs to `text_encoder`.
        text_encoder: Where the `lora_state_dict` is to be set.
    """

    text_encoder_state_dict = {
        f'{k.replace(prefix, "")}': v for k, v in lora_state_dict.items() if k.startswith(prefix)
    }
    text_encoder_state_dict = convert_state_dict_to_peft(convert_state_dict_to_diffusers(text_encoder_state_dict))
    set_peft_model_state_dict(text_encoder, text_encoder_state_dict, adapter_name="default")


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
        default=None,
        required=True,
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
        default=None,
        required=True,
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
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
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
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreambooth-inpaint-model",
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
        "--checkpointing_steps",
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
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
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
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

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms_resize_and_crop(class_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
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
        self.instance_image_list = [os.path.join(instance_image_dir, image_file) for image_file in self.instance_image_captions.keys()]
        self.class_image_captions = self.caption_file_reader(class_image_captions_file)
        self.class_image_list = [os.path.join(class_images_dir, image_file) for image_file in self.class_image_captions.keys()]
        if len(self.class_image_list) > len(self.instance_image_list):
            self.class_image_list = random.sample(self.class_image_list, len(self.instance_image_list) - 1)
        elif len(self.class_image_list) < len(self.instance_image_list):
            required_samples_count = len(self.instance_image_list) - len(self.class_image_list)
            while required_samples_count > 0:
                if required_samples_count > len(self.class_image_list):
                    class_image_list = random.sample(self.class_image_list, len(self.class_image_list) - 1)
                    required_samples_count -= (len(self.class_image_list) - 1)
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
            self.instance_image_captions[image_name],
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



class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
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

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size, num_workers=1
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)
            transform_to_pil = transforms.ToPILImage()
            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                bsz = len(example["prompt"])
                fake_images = torch.rand((3, args.resolution, args.resolution))
                transform_to_pil = transforms.ToPILImage()
                fake_pil_images = transform_to_pil(fake_images)

                fake_mask = random_mask((args.resolution, args.resolution), ratio=1, mask_full_image=True)

                images = pipeline(prompt=example["prompt"], mask_image=fake_mask, image=fake_pil_images).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    # lora_attn_procs = {}
    # for name in unet.attn_processors.keys():
    #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #     if name.startswith("mid_block"):
    #         hidden_size = unet.config.block_out_channels[-1]
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")])
    #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet.config.block_out_channels[block_id]

    #     lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    # unet.set_attn_processor(lora_attn_procs)
    # lora_layers = AttnProcsLayers(unet.attn_processors)

    # accelerator.register_for_checkpointing(lora_layers)
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet.add_adapter(unet_lora_config)

    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder))):
                    text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.append(text_encoder_)

            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


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

    # optimizer = optimizer_class(
    #     lora_layers.parameters(),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )
    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.instance_images_mask_dir is None:
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
        )

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

    # Prepare everything with our `accelerator`.
    # lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     lora_layers, optimizer, train_dataloader, lr_scheduler
    # )
        # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    # accelerator.register_for_checkpointing(lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-inpaint-lora", config=vars(args))

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
        if args.train_text_encoder:
            text_encoder.train()
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
                        torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                        for mask in masks
                    ]
                ).to(dtype=weight_dtype)
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
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Save the lora layers
    # if accelerator.is_main_process:
    #     unet = unet.to(torch.float32)
    #     unet.save_attn_procs(args.output_dir)

    #     if args.push_to_hub:
    #         upload_folder(
    #             repo_id=repo_id,
    #             folder_path=args.output_dir,
    #             commit_message="End of training",
    #             ignore_patterns=["step_*", "epoch_*"],
    #         )
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet = unet.to(torch.float32)

        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_text_encoder:
            text_encoder = unwrap_model(text_encoder)
            text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
        else:
            text_encoder_state_dict = None

        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_state_dict,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
