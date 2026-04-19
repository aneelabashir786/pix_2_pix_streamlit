import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# ── Model Definitions (same architecture as training) ─────────────────────────
class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize: 
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        if dropout: 
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): 
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_c), 
                  nn.ReLU(True)]
        if dropout: 
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip): 
        return torch.cat([self.model(x), skip], dim=1)

class UNetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super().__init__()
        self.d1 = UNetDown(in_c, 64,  normalize=False)
        self.d2 = UNetDown(64,  128)
        self.d3 = UNetDown(128, 256)
        self.d4 = UNetDown(256, 512)
        self.d5 = UNetDown(512, 512)
        self.d6 = UNetDown(512, 512)
        self.d7 = UNetDown(512, 512)
        self.d8 = UNetDown(512, 512, normalize=False)
        self.u1 = UNetUp(512,  512, dropout=0.5)
        self.u2 = UNetUp(1024, 512, dropout=0.5)
        self.u3 = UNetUp(1024, 512, dropout=0.5)
        self.u4 = UNetUp(1024, 512)
        self.u5 = UNetUp(1024, 256)
        self.u6 = UNetUp(512,  128)
        self.u7 = UNetUp(256,  64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_c, 4, 2, 1), 
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        
        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        
        return self.final(u7)

# ── Helper functions ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_url, device):
    """Load the generator model from Hugging Face"""
    model = UNetGenerator().to(device)
    
    # Download weights from Hugging Face
    with st.spinner('Downloading model weights...'):
        response = requests.get(model_url)
        if response.status_code == 200:
            state_dict = torch.load(BytesIO(response.content), map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        else:
            st.error(f"Failed to download model from {model_url}")
            return None

def preprocess_image(image, target_size=256):
    """Preprocess input image for the model"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to tensor and normalize to [-1, 1]
    img_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] and then to PIL Image"""
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def generate_photo(model, sketch_tensor, device):
    """Generate photo from sketch using the model"""
    with torch.no_grad():
        sketch_tensor = sketch_tensor.to(device)
        generated = model(sketch_tensor)
        generated = generated.cpu()
    return generated

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pix2Pix: Doodle to Real Image",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .info-text {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎨 Pix2Pix: Doodle to Real Image</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Transform your sketches and doodles into realistic images using GANs</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Instructions")
    st.markdown("""
    1. **Upload** a sketch/doodle image
    2. **Click** the 'Generate' button
    3. **View** the AI-generated realistic image
    
    ### 💡 Tips:
    - Use simple sketches with clear outlines
    - Best results with face sketches or anime-style drawings
    - Image will be resized to 256×256 pixels
    - Upload PNG, JPG, or JPEG files
    
    ### 🎯 Model Info:
    - Architecture: U-Net Generator with Skip Connections
    - Training: Pix2Pix GAN (Adversarial + L1 Loss)
    - Dataset: CUHK Face Sketch + Anime Sketch-Color pairs
    - Training time: 50 epochs
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("SSIM", "0.72", "Structural Similarity")
    with col2:
        st.metric("PSNR", "18.5 dB", "Peak Signal-to-Noise")
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit & PyTorch")

# Main content area
col1, col2 = st.columns(2)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.info(f"🖥️ Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Model URLs
G_MODEL_URL = "https://huggingface.co/aneelaBashir22f3414/pix2pix_doodle_to_real_image/resolve/main/p2p_G_ep50.pth"
D_MODEL_URL = "https://huggingface.co/aneelaBashir22f3414/pix2pix_doodle_to_real_image/resolve/main/p2p_D_ep50.pth"

# Load model
model = load_model(G_MODEL_URL, device)
if model is None:
    st.error("Failed to load model. Please check the model URL.")
    st.stop()

# Input section
with col1:
    st.markdown("### ✏️ Input Sketch")
    uploaded_file = st.file_uploader(
        "Choose a sketch image...", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
        help="Upload your sketch or doodle here"
    )
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Sketch", use_container_width=True)
        
        # Display preprocessing info
        with st.expander("📐 Image Info"):
            st.write(f"Original Size: {input_image.size}")
            st.write(f"Mode: {input_image.mode}")
            st.write("Will be resized to 256×256 for the model")
    
    # Example images
    with st.expander("🎨 Try these examples"):
        st.markdown("Upload a sketch similar to these for best results:")
        st.markdown("- Face sketches (front-facing)")
        st.markdown("- Anime character doodles")
        st.markdown("- Simple object outlines")
        st.markdown("- Black & white line drawings")

# Output section
with col2:
    st.markdown("### 🖼️ Generated Image")
    
    if uploaded_file is not None:
        if st.button("🚀 Generate Realistic Image", type="primary", use_container_width=True):
            with st.spinner("🧠 Processing your sketch... This may take a few seconds."):
                # Preprocess
                input_tensor = preprocess_image(input_image)
                
                # Generate
                generated_tensor = generate_photo(model, input_tensor, device)
                
                # Convert to PIL
                output_image = denormalize(generated_tensor)
                
                # Display
                st.image(output_image, caption="AI-Generated Image", use_container_width=True)
                
                # Download button
                buf = BytesIO()
                output_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Result",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Metrics for this specific generation (optional)
                st.success("✨ Generation completed successfully!")
    else:
        st.info("👈 Upload a sketch to get started!")
        # Placeholder
        placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
        placeholder[:] = [240, 240, 240]
        st.image(placeholder, caption="Your generated image will appear here", use_container_width=True)

# Compare section (if both input and output are available)
if uploaded_file is not None and 'output_image' in locals():
    st.markdown("---")
    st.markdown("### 🔍 Before vs After")
    
    col1, col2, col3 = st.columns([1, 0.2, 1])
    with col1:
        st.markdown("**Original Sketch**")
        st.image(input_image, use_container_width=True)
    with col2:
        st.markdown("<h3 style='text-align: center; margin-top: 50%;'>→</h3>", unsafe_allow_html=True)
    with col3:
        st.markdown("**Generated Photo**")
        st.image(output_image, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>This model was trained on face sketches and anime-style drawings. 
    Results may vary depending on the complexity and style of your input sketch.</p>
    <p>🔬 Model trained with Pix2Pix architecture using U-Net generator and PatchGAN discriminator</p>
</div>
""", unsafe_allow_html=True)
