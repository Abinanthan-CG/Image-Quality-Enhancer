import streamlit as st
import torch
import cv2
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO

# We can now import these directly because requirements.txt is configured correctly
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- App UI Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AI Image Enhancer",
    page_icon="✨"
)

# --- NEW: Custom CSS to replicate Hugging Face's aesthetic ---
# This injects CSS to center the main content block and give it a max-width
st.markdown("""
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .st-emotion-cache-16txtl3 {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Model & File Configuration ---
MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
MODEL_NAME = 'RealESRGAN_x4plus.pth'

# --- NEW: Example Images ---
EXAMPLE_IMAGES = {
    "Baboon": "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/assets/baboon.png",
    "Comic": "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/assets/comic.png",
    "Cat": "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/assets/cat_lq.jpg",
}

# --- Cached Functions for Model Loading and Downloads ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Real-ESRGAN model. This is cached for performance."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    device = torch.device("cpu")
    upsampler = RealESRGANer(
        scale=4, model_path=MODEL_NAME, model=model,
        device=device, tile=0, tile_pad=10, pre_pad=0, half=False
    )
    return upsampler

@st.cache_data
def download_file(url, file_name):
    """Downloads a file from a URL, used for both the model and examples."""
    if not os.path.exists(file_name):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_name, 'wb') as f:
                for data in response.iter_content(1024):
                    f.write(data)
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading file: {e}")
            return None
    return file_name

# --- Main App Interface ---
st.title("✨ Real-ESRGAN Image Super-Resolution")
st.markdown(
    "This demo uses **Real-ESRGAN** to enhance low-resolution images. "
    "Upload your own image or try one of the examples below to see the AI in action."
)
st.write("---")

# Main app logic starts here
download_file(MODEL_URL, MODEL_NAME)
upsampler = load_model()

# --- NEW: Clickable Examples Section ---
st.subheader("Try an Example")
example_cols = st.columns(len(EXAMPLE_IMAGES))
# This dictionary will hold the image bytes for processing
image_to_process = None

for col, (name, url) in zip(example_cols, EXAMPLE_IMAGES.items()):
    with col:
        st.image(url, use_column_width=True)
        if st.button(f"Use {name}", use_container_width=True):
            # Download the example image and prepare it for processing
            st.toast(f"Loading '{name}' example...")
            image_bytes = requests.get(url).content
            image_to_process = image_bytes # Store the bytes

# --- File Uploader ---
st.subheader("Upload Your Own Image")
uploaded_file = st.file_uploader(
    "Choose a low-resolution JPG or PNG file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image_to_process = uploaded_file.getvalue()

# --- Processing and Displaying Results ---
if image_to_process:
    st.write("---")
    st.subheader("Results")

    # Convert image bytes to a NumPy array for OpenCV
    file_bytes = np.asarray(bytearray(image_to_process), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)

    with st.spinner('The AI is working its magic...'):
        output_image, _ = upsampler.enhance(input_image, outscale=4)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original Image")
        st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.caption("Enhanced Image")
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Download Button
        result_bytes = cv2.imencode('.png', output_image)[1].tobytes()
        file_name = "enhanced_image.png"
        if uploaded_file: # Use original filename if available
            file_name = f"enhanced_{uploaded_file.name}"
        
        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=result_bytes,
            file_name=file_name,
            mime="image/png"
        )

# --- NEW: Footer Section ---
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
