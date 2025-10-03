import streamlit as st
import torch
import timm
from peft import LoraConfig, get_peft_model
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
import copy

# --- Grad-CAM Imports ---
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Page Configuration (Set this at the top) ---
st.set_page_config(
    page_title="ViT for Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

# --- Model and Configuration Loading ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model_and_config():
    # Define paths and constants
    MODEL_NAME = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
    # NOTE: You might need to adjust the path depending on where you place the files
    PARAMS_PATH = 'best_centralized_params.csv'
    MODEL_PATH = 'final_fl_lora_model.pth'
    
    # Define NUM_CLASSES (hardcoded for simplicity, can be dynamic)
    NUM_CLASSES = 2
    class_names = ['benign', 'malignant']

    # Load the best hyperparameters
    best_params = pd.read_csv(PARAMS_PATH).to_dict('records')[0]

    # Rebuild the PEFT model structure
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    config = LoraConfig(
        r=best_params['lora_r'],
        lora_alpha=best_params['lora_alpha'],
        target_modules=['qkv'],
        bias="none"
    )
    final_model = get_peft_model(model, config)

    # Load the saved state dictionary
    # Use map_location=torch.device('cpu') to ensure it runs on any machine
    final_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    final_model.eval()
    
    print("Model loaded successfully!")
    return final_model, class_names

# --- Image Transformations ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# --- Grad-CAM Helper Function ---
def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

# --- Main App ---
st.title("🔬 AI for Breast Cancer Histology Classification")
st.markdown(
    "This application uses a Vision Transformer (ViT) model, fine-tuned with LoRA and trained in a simulated "
    "Federated Learning environment, to classify breast cancer histology images as benign or malignant."
)
st.markdown("---")

# Load the model
with st.spinner('Loading the AI model, this may take a moment...'):
    model, class_names = load_model_and_config()

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a histology image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # --- Display Uploaded Image ---
    st.header("Uploaded Image")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Your uploaded image.', use_column_width=True)
    st.markdown("---")

    # --- Classification ---
    if st.button('Classify Image', type="primary"):
        with st.spinner('Analyzing...'):
            # Preprocess the image
            input_tensor = val_test_transform(image).unsqueeze(0)

            # Get model prediction
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            pred_label_name = class_names[pred_idx.item()]
            confidence_score = confidence.item()

            # --- Grad-CAM Analysis ---
            target_layers = [model.base_model.model.blocks[-1].attn.qkv]
            cam = LayerCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit)
            targets = [ClassifierOutputTarget(pred_idx.item())]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            # Overlay heatmap
            resized_img_for_viz = image.resize((224, 224))
            rgb_img_float = np.array(resized_img_for_viz) / 255.0
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

            # --- Display Results ---
            st.header("Analysis Results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", pred_label_name.capitalize())
                st.write("Confidence:")
                st.progress(confidence_score)
                st.write(f"{confidence_score:.2%}")

            with col2:
                st.image(visualization, caption='Model Explanation (Grad-CAM)')

st.sidebar.title("About the Project")
st.sidebar.info(
    "**Model:** Vision Transformer (ViT-Small)\n\n"
    "**Methodology:**\n"
    "- Transfer Learning\n"
    "- Federated Learning (Simulated)\n"
    "- PEFT (LoRA)\n\n"
    "This app demonstrates how a large AI model can be trained on private, decentralized data in a communication-efficient way."
)
