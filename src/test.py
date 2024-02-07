import json
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer
# from src.sd_inpainting_finetuning import TrainDataset, prepare_mask_and_masked_image, random_mask
# from sd_inpainting_finetuning import TrainDataset, prepare_mask_and_masked_image, random_mask


if __name__== "__main1__":
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")

    train_dataset = TrainDataset(instance_image_captions_file=r"C:\Users\smanjun3\Desktop\FashionXchange\2667_IMAGES_dict.json",
                                 instance_image_dir=r"C:\Users\smanjun3\Desktop\FashionXchange\IMAGES",
                                 image_size=512,
                                 tokenizer=tokenizer,
                                 center_crop=False,
                                 class_image_dir=None,
                                 class_image_prompts_file=None,
                                 class_sample_generated_prompts=None)
    
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if False: #args.with_prior_preservation:
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

        if False: #args.with_prior_preservation:
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

    train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    for i, example in tqdm(enumerate(train_dataloader)):
        if example["pixel_values"].shape != torch.Size([4, 3, 512, 512]):
            print()


if __name__ == "__main2__":
    mask_path = r"C:\Users\sumuk\Downloads\content_mask.npy"
    image_path = r"C:\Users\sumuk\Downloads\MEN-Denim-id_00000089-01_7_additional.jpg"
    mask = np.load(mask_path)
    image = np.array(Image.open(image_path))
    ones = np.where(mask == True)
    y_max, y_min, x_max, x_min = max(ones[0]), min(ones[0]), max(ones[1]), min(ones[1])
    increase_or_decrease = [0, 1]
    up_down_left_right = [0, 1, 2, 3]
    choice_increase_or_decrease = np.random.choice(increase_or_decrease)
    choice_udlr = np.random.choice(up_down_left_right)
    if choice_udlr == 0 or choice_udlr == 1:
        increase_or_decrease_quantity = np.random.randint(10, (y_max - y_min) // 2)
    elif choice_udlr == 2 or choice_udlr == 3:
        increase_or_decrease_quantity = np.random.randint(10, (x_max - x_min) // 2)
    if choice_increase_or_decrease == 0:
        if choice_udlr == 0:
            increase_or_decrease_quantity = min(increase_or_decrease_quantity, y_min)
            mask[y_min - increase_or_decrease_quantity: y_min, x_min: x_max + 1] = True
        elif choice_udlr == 1:
            increase_or_decrease_quantity = min(increase_or_decrease_quantity, mask.shape[0] - y_max)
            mask[y_max: y_max + increase_or_decrease_quantity, x_min: x_max + 1] = True
        elif choice_udlr == 2:
            increase_or_decrease_quantity = min(x_min, increase_or_decrease_quantity)
            mask[y_min: y_max + 1, x_min - increase_or_decrease_quantity: x_min] = True
        else:
            increase_or_decrease_quantity = min(mask.shape[1] - x_max, increase_or_decrease_quantity)
            mask[y_min: y_max + 1, x_max: x_max + increase_or_decrease_quantity] = True
    elif choice_increase_or_decrease == 1:
        if choice_udlr == 0:
            mask[y_min: y_min + increase_or_decrease_quantity, x_min: x_max + 1] = False
        elif choice_udlr == 1:
            mask[y_max: y_max - increase_or_decrease_quantity, x_min: x_max + 1] = False
        elif choice_udlr == 2:
            mask[y_min: y_max + 1, x_min: x_min + increase_or_decrease_quantity] = False
        else:
            mask[y_min: y_max + 1, x_max - increase_or_decrease_quantity: x_max] = False
    print("")


if __name__ == "__main3__":
    captions_file_path = r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\captions.json"
    masks_folder = r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\Masks"
    save_path = r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\masked_image_captions.json"
    with open(captions_file_path, "r") as json_file:
        image_vs_captions = json.load(json_file)
    masked_image_list = os.listdir(masks_folder)
    masked_image_names = [os.path.basename(i) for i in masked_image_list]
    masked_image_names = [i + ".jpg" for i in masked_image_names]
    masked_image_vs_captions = dict()
    for masked_image_name in masked_image_names:
        masked_image_vs_captions[masked_image_name] = image_vs_captions[masked_image_name]
    with open(save_path, "w") as json_file:
        json.dump(masked_image_vs_captions, json_file)


if __name__ == "__main__":
    import os
    from PIL import Image
    import numpy as np
    import torch
    import json
    from glob import glob
    from torch.utils.data import Dataset
    from torchvision import transforms
    from transformers import CLIPTokenizer

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
        
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")

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
        

    def random_perturb_mask(mask):
        ones = np.where(mask > 0)
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
    

    train_dataset = TargetedMaskingDataset(instance_image_captions_file=r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\masked_image_captions.json", 
                                            instance_image_dir=r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\images",
                                            instance_images_mask_dir=r"C:\Users\smanjun3\Desktop\FashionXchange\text2human\Masks", 
                                            tokenizer=tokenizer,
                                            image_size=512)

    print(f"Training dataset of {len(train_dataset)} images loaded")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    for i, example in enumerate(train_dataloader):
        print("")