import streamlit as st
import torch
import cv2
import numpy as np
import os
import requests
from tqdm import tqdm

# We can now import these directly because requirements.txt will install compatible versions
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- App UI Configuration ---
st.set_page_config(layout="wide", page_title="AI Image Enhancer")
st.title("🖼️ AI-Powered Image Super-Resolution")
st.info("Upload a low-resolution image to see it enhanced in real-time using a pre-trained Real-ESRGAN model.")

# --- Model & File Configuration ---
MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
MODEL_NAME = 'RealESRGAN_x4plus.pth'

# --- Function to download the model file manually ---
def download_model(url, model_name):
    if not os.path.exists(model_name):
        st.info(f"Downloading the AI model ({model_name})... This may take a moment.")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = st.progress(0)
            with open(model_name, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.progress(f.tell() / total_size)
            progress_bar.empty()
            st.success("Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.stop()

# --- Model Loading (with caching) ---
@st.cache_resource
def load_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    device = torch.device("cpu")
    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_NAME,
        model=model, device=device, tile=0, tile_pad=10, pre_pad=0, half=False)
    return upsampler

# --- Main App Logic ---
download_model(MODEL_URL, MODEL_NAME)
upsampler = load_model()

# --- Image Uploader and Processing Logic ---
uploaded_file = st.file_uploader("Choose an image to enhance...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)
    st.write("---")
    with st.spinner('AI is enhancing your image... This can take up to 30 seconds.'):
        output_image, _ = upsampler.enhance(input_image, outscale=4)
    st.header("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        # --- THIS LINE IS FIXED ---
        st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), width="stretch")
    with col2:
        st.subheader("Enhanced Image")
        # --- THIS LINE IS FIXED ---
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), width="stretch")
else:
    st.warning("Please upload an image to get started.")
