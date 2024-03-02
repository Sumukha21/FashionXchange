import os
from PIL import Image
import numpy as np
import torch
import random
import json
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils.utils import text_file_reader


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

