import streamlit as st
import torch
import cv2
import numpy as np
import os
import requests
from PIL import Image

# Import the necessary model components
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- App UI Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AI Image Enhancer",
    page_icon="✨"
)

# --- Custom CSS for the Hugging Face Aesthetic ---
st.markdown("""
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Model & File Configuration ---
MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
MODEL_NAME = 'RealESRGAN_x4plus.pth'

# --- Cached Functions for Model Loading and Downloading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Real-ESRGAN model."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    device = torch.device("cpu")
    upsampler = RealESRGANer(
        scale=4, model_path=MODEL_NAME, model=model,
        device=device, tile=0, tile_pad=10, pre_pad=0, half=False
    )
    return upsampler

@st.cache_data
def download_model(url, file_name):
    """Downloads the model file from a URL if it doesn't exist."""
    if not os.path.exists(file_name):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_name, 'wb') as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            return None
    return file_name

# --- Main App Interface ---
st.title("✨ Real-ESRGAN Image Super-Resolution")
st.markdown(
    "Upload a JPG, JPEG, or PNG file to see the AI increase its resolution by 4x. "
    "The model will be loaded after you upload an image."
)
st.write("---")

# --- File Uploader and Main Processing Logic ---
st.subheader("Upload Your Image")
uploaded_file = st.file_uploader(
    "Choose a low-resolution image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # ---- THIS IS THE NEW "LAZY LOADING" LOGIC ----
    with st.spinner('Loading AI model and enhancing your image... Please wait.'):
        # 1. Ensure the model file is downloaded
        download_model(MODEL_URL, MODEL_NAME)
        
        # 2. Load the model (this will be cached after the first time)
        upsampler = load_model()

        # 3. Process the uploaded file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)
        
        # 4. Enhance the image
        output_image, _ = upsampler.enhance(input_image, outscale=4)

    # Display results
    st.write("---")
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original Image")
        st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR_RGB), use_column_width=True)
    with col2:
        st.caption("Enhanced Image")
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR_RGB), use_column_width=True)

        # Download Button
        result_bytes = cv2.imencode('.png', output_image)[1].tobytes()
        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=result_bytes,
            file_name=f"enhanced_{uploaded_file.name}",
            mime="image/png"
        )
else:
    st.info("Please upload an image to get started.")

# --- Footer Section ---
st.write("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.9em;">
        <p><strong>Model:</strong> <a href="https://github.com/xinntao/Real-ESRGAN" target="_blank">Real-ESRGAN</a> by xinntao</p>
        <p><strong>App built with:</strong> <a href="https://streamlit.io" target="_blank">Streamlit</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
