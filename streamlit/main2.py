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

    sam_predictor = load_sam_model()

    # Stable Diffusion
    @st.cache_resource
    def load_pipeline():
        return StableDiffusionInpaintPipeline.from_pretrained(r"C:\Users\sumuk\OneDrive\Desktop\GitHub\FashionXchange\checkpoint-82752").to(device)

    pipeline = load_pipeline()

    # Grounding DINO
    @st.cache_resource
    def load_gdino():
        return load_model(groundingdino_model_path, groundingdino_weights_path)

    groundingdino_model = load_gdino()


    if "data_submitted" not in st.session_state:
        st.session_state.data_submitted = False

    if "person_selected" not in st.session_state:
        st.session_state.person_selected = False

    other_field = st.empty()
        
    with st.sidebar:
        st.write("Upload file")
        uploaded_file = st.file_uploader("Choose a file")
        looking_for= st.selectbox('Select dress areas to modify:', ('Clothes, arms, legs', 'Shirt, arms', 'Pant, legs'))
        input_string = st.text_input("Enter prompt to make changes:", "A man wearing a blue shirt")

        if st.button("Submit"):
            if uploaded_file is not None and looking_for is not None and input_string is not None:
                st.session_state.data_submitted = True

    if st.session_state.data_submitted:
        src, img = load_image(uploaded_file)
        imageLocation = st.empty()
        imageLocation.image(src)

        with st.spinner('Searching for people in the image...'):
            boxes_og, logits, phrases = dino_person_prediction(img, groundingdino_model, device)
            h, w, _ = src.shape
            boxes = (boxes_og * torch.Tensor([w, h, w, h])).to(torch.int16)
            boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            boxes = np.array(boxes, dtype=np.int16)
        
        imageLocation.empty()

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
                cols = st.columns(columns)  #cols = columns_layout.columns(columns) 
                for idx2 in range(columns): 
                    idx_cur = idx * columns + idx2
                    if idx_cur >= total_image_count:
                        continue
                    person = Image.fromarray(people_list[idx_cur])
                    p = cols[idx2].image(person.resize((100, 200), resample=1), use_column_width=True)
                    people_placeholders.append(p)
            options = [f"Person {i}" for i in range(total_image_count)]
            selected_option = other_field.selectbox("Select an option", ["Please Select", *options])
            if selected_option != "Please Select":
                selected_person = int(selected_option.split(" ")[1])
                other_field.empty()
                for p_i in people_placeholders:
                    p_i.empty()
                # st.write("Selected person: ", selected_person)
                imageLocation.image(Image.fromarray(people_list[selected_person]).resize((100, 200), 1))
                person = people_list[selected_person]
                selected_person_box = boxes[selected_person]
                st.session_state.person_selected = True
        else:
            box = boxes[0]
            person = src[box[1]: box[3], box[0]: box[2], :]
            # st.write("Selected person: ")
            imageLocation.image(person)
            st.session_state.person_selected = True 
        
        if st.session_state.person_selected:
            with st.spinner('Making the requesting outfit modification'):
                attribute_boxes, attribute_logits, attribute_phrases = get_outfit_in_person(person, "clothes", groundingdino_model, device)
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
                            mask_image=Image.fromarray(masks[0][0].numpy()).resize((512, 512))
                            ).images[0]
                edited_image = edited_image.resize((person.shape[1], person.shape[0]))
                src = np.array(src)
                src[selected_person_box[1]: selected_person_box[3], selected_person_box[0]: selected_person_box[2]] = edited_image
                imageLocation.empty()
                imageLocation.image(src)


# # Main Streamlit app
# def main():
#     # Check if data is already submitted
#     if "data_submitted" not in st.session_state:
#         st.session_state.data_submitted = False
#     other_field = st.empty()
#     # Sidebar with file uploader and submit button
#     with st.sidebar:
#         st.write("Upload file")
#         uploaded_file = st.file_uploader("Choose a file")

#         # Check if submit button is clicked
#         if st.button("Submit"):
#             if uploaded_file is not None:
#                 st.session_state.data_submitted = True

#     # Main content area
#     if st.session_state.data_submitted:
#         imageLocation = st.empty()
#         # Display the uploaded image
#         img = Image.open(uploaded_file)
#         imageLocation.image(img, caption='Uploaded Image', use_column_width=True)
                
#         # Dropdown for user interaction
#         selected_option = other_field.selectbox("Select an option", ["Please Select", "Option 1", "Option 2", "Option 3"])

#         # Check if an option is selected
#         if selected_option != "Please Select":
#             other_field.empty()
#             # Display another image
#             st.write("People displayed")
#             images = [img for i in range(12)]
#             imageLocation.empty()
#             for idx in range(3):
#                 cols = st.columns(4) 
#                 for idx2 in range(4): 
#                     cols[idx2].image(images[(idx + 1) * idx2], use_column_width=True)
        

# Run the main function
if __name__ == "__main__":
    main()
