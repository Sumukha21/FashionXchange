# FashionXchange
Have you ever had a photo of your's where you had not dressed for the occation or had overdressed?
Worry not! With our application (in development) you can modify  your clothes with just text instructions.

![FashionXchange backend dataflow](dependencies\workflow.png)


Our pipeline for modifying outfits uses GroundingDINO, Segment ANything Model and Stable Diffusion.
We are currently finetuning our model to make it accurate in modifying outfits. Stay tuned!

Reference:
1. https://arxiv.org/pdf/2112.10752.pdf
2. https://github.com/huggingface/diffusers/blob/main/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py
3. https://github.com/orbitalsonic/Fashion-Dataset-Images-Western-Dress/tree/master/WesternDress_Images
4. https://huggingface.co/datasets/nlphuji/flickr30k/tree/main
