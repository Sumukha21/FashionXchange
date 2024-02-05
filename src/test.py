import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer
# from src.sd_inpainting_finetuning import TrainDataset, prepare_mask_and_masked_image, random_mask
from sd_inpainting_finetuning import TrainDataset, prepare_mask_and_masked_image, random_mask


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


if __name__ == "__main__":
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

    