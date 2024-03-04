import torch
import streamlit as st
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline
from groundingdino.util.inference import load_model, load_image, annotate
from utils import dino_person_prediction, find_person_in_image, get_outfit_in_person, show_mask, transform_boxes
from torchvision.ops import box_convert

def main():
    st.title('FashionXchange')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Paths
    sam_checkpoint_path = "GroundingDINO\weights\sam_vit_h_4b8939.pth"
    groundingdino_model_path = "GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
    groundingdino_weights_path = "GroundingDINO\weights\groundingdino_swint_ogc.pth"
    # SAM Parameters
    model_type = "vit_h"
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device=device)

    # Load SAM Model
    @st.cache_resource
    def load_sam_model():
        return SamPredictor(sam_model)

    sam_predictor = load_sam_model()

    # Load Stable Diffusion
    @st.cache_resource
    def load_pipeline():
        return StableDiffusionInpaintPipeline.from_pretrained("FashionXchange/FashionXchange_diffuser").to(device)

    pipeline = load_pipeline()

    # Load Grounding DINO
    @st.cache_resource
    def load_gdino():
        return load_model(groundingdino_model_path, groundingdino_weights_path)

    groundingdino_model = load_gdino()

    # Initialize streamlit session state variables

    if "image_submitted" not in st.session_state:
        st.session_state.image_submitted = False

    if "data_submitted" not in st.session_state:
        st.session_state.data_submitted = False

    if "person_selected" not in st.session_state:
        st.session_state.person_selected = False

    if "allow_modifications" not in st.session_state:
        st.session_state.allow_modifications = False
    
    if "original_image_displayed" not in st.session_state:
        st.session_state.original_image_displayed = False

    if "person_images_shown" not in st.session_state:
        st.session_state.person_images_shown = False

    if "boxes" not in st.session_state:
        st.session_state.boxes = 0  

    other_field = st.empty()
    side_bar=st.sidebar
    imageLocation = st.empty()

    # Sidebar elements   
    with side_bar:
        st.write("Upload file")
        uploaded_file = st.file_uploader("Choose a file")
        if st.button("Upload Image"):
            if uploaded_file is not None:
                st.session_state.image_submitted=True


    if st.session_state.image_submitted:
        src, img = load_image(uploaded_file)
       
        if st.session_state.image_submitted and not st.session_state.original_image_displayed:
            imageLocation.image(src)

            with st.spinner('Searching for people in the image...'):
                boxes_og, logits, phrases = dino_person_prediction(img, groundingdino_model, device)
                h, w, _ = src.shape
                ss_boxes = (boxes_og * torch.Tensor([w, h, w, h])).to(torch.int16)
                ss_boxes = box_convert(boxes=ss_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                ss_boxes = np.array(ss_boxes, dtype=np.int16)
                st.session_state.boxes = ss_boxes
        
        imageLocation.empty()
        st.session_state.original_image_displayed = True
        boxes = st.session_state.boxes

        if len(boxes) > 1:
            people_list = []
            for i in range(len(boxes)):
                box = [int(j) for j in boxes[i]]
                person_i = src[box[1]: box[3], box[0]: box[2], :]
                people_list.append(person_i)    
            total_image_count = len(people_list)
            columns = min(len(people_list), 4)
            if len(people_list) % columns == 0:
                row = len(people_list) // columns
            else:
                row = (len(people_list) // columns) + 1
            people_placeholders = []
            for idx in range(row):
                cols = st.columns(columns) 
                for idx2 in range(columns): 
                    idx_cur = idx * columns + idx2
                    if idx_cur >= total_image_count:
                        continue
                    person = Image.fromarray(people_list[idx_cur])
                    if not st.session_state.person_selected and not st.session_state.person_images_shown:
                        p = cols[idx2].image(person.resize((100, 200), resample=1), use_column_width=True)
                        people_placeholders.append(p)
            options = [f"Person {i}" for i in range(total_image_count)]
            selected_option = other_field.selectbox("Select an option", ["Please Select", *options])
            if selected_option != "Please Select":
                selected_person = int(selected_option.split(" ")[1])
                other_field.empty()
                for p_i in people_placeholders:
                    p_i.empty()
                if not st.session_state.data_submitted:   
                    imageLocation.image(Image.fromarray(people_list[selected_person]).resize((300, 400), 1))
                person = people_list[selected_person]
                selected_person_box = boxes[selected_person]
                st.session_state.person_selected = True
               
        else:
            box = boxes[0]
            selected_person_box = box
            person = src[box[1]: box[3], box[0]: box[2], :]
            if not st.session_state.person_selected and not st.session_state.person_images_shown:
                imageLocation.image(person)
            st.session_state.person_selected = True 
        
        st.session_state.person_images_shown = True

    with side_bar:
        if st.session_state.image_submitted and st.session_state.person_selected:
            st.write("Select modifications after person is selected")
            looking_for= st.selectbox('Select dress areas to modify:', ('Clothes, arms, legs', 'Shirt, arms', 'Pant, legs'))
            input_string = st.text_input("Enter prompt to make changes:", "A man wearing a blue shirt")

            if st.button("Submit"):
                if uploaded_file is not None and looking_for is not None and input_string is not None:
                    st.session_state.data_submitted = True
        
    if st.session_state.person_selected and st.session_state.data_submitted:

        with st.spinner('Making the requested outfit modification'):
            attribute_boxes, attribute_logits, attribute_phrases = get_outfit_in_person(person, looking_for, groundingdino_model, device)
            attribute_boxes_tr = transform_boxes(sam_predictor, attribute_boxes, person, device)

            sam_predictor.set_image(np.array(person))
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=torch.Tensor(attribute_boxes_tr),
                multimask_output=False,
            )
            masks = torch.any(masks, dim=0, keepdim=True)

            edited_image = pipeline(prompt=input_string,
                        image=Image.fromarray(person).resize((512, 512)),
                        mask_image=Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))
                        ).images[0]
            edited_image = edited_image.resize((person.shape[1], person.shape[0]))
            src = np.array(src)
            src[selected_person_box[1]: selected_person_box[3], selected_person_box[0]: selected_person_box[2]] = edited_image
            imageLocation.empty()
            imageLocation.image(src)

# Run the main function
if __name__ == "__main__":
    main()
