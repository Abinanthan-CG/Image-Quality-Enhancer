import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# We can now import these directly because requirements.txt will install compatible versions
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- App UI Configuration ---
st.set_page_config(layout="wide", page_title="AI Image Enhancer")
st.title("🖼️ AI-Powered Image Super-Resolution")
st.info("Upload a low-resolution image to see it enhanced in real-time using a pre-trained Real-ESRGAN model.")

# --- Model Loading (with caching) ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Real-ESRGAN model."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # Force the model to run on CPU. This is more reliable for free deployment services.
    device = torch.device("cpu")
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model, device=device, tile=0, tile_pad=10, pre_pad=0, half=False)
    return upsampler

with st.spinner('Loading AI model... This may take a minute on the first startup.'):
    upsampler = load_model()
st.success("AI Model Loaded!")

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
        st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Enhanced Image")
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_column_width=True)
else:
    st.warning("Please upload an image to get started.")
