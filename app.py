"""
LumaCam: Edge-Native Real-Time Low-Light Camera Demo (Streamlit)
==============================================================
Interactive dashboard demonstrating:
1. Dynamic Exposure Slider (FiLM latent conditioning vector alpha)
2. Dual-mode input: Synthetic physical Bayer Poisson-Gaussian simulation or real .ARW raw files
3. Four-way Computational Photography comparison:
   - Underexposed Raw Bayer Sensor preview
   - Linear Digital ISO Gain (illustrating noise blow-up)
   - LumaCam Attention U-Net GAN Enhancement
   - LumaCam + Color Science (AWB & Saturation)
4. Explainable AI: Live 4-stage Spatial Attention Gate heatmaps showing shot noise suppression
5. Real-time inference latency and performance metrics (FPS, PSNR)
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageFilter

# Ensure repository root is in python search path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.networks.generator import ExposureConditionedAttentionUNet
from model.datasets.sid_dataset import DummySIDDataset, SIDSonyDataset
from model.datasets.raw_processing import pack_raw_bayer_np
from model.postprocess.color_science import apply_color_pipeline

st.set_page_config(
    page_title="LumaCam | Low-Light AI Camera Demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_lumacam_model(checkpoint_path_str: str):
    """Loads and caches the PyTorch generator on CUDA or CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = ExposureConditionedAttentionUNet(
        in_channels=4,
        out_channels=3,
        base_channels=32,
        cond_dim=1,
        use_pixel_shuffle=True,
    ).to(device)

    ckpt_path = Path(checkpoint_path_str)
    loaded = False
    meta = {}
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict):
                if "generator_state_dict" in ckpt:
                    generator.load_state_dict(ckpt["generator_state_dict"], strict=False)
                elif "generator_state" in ckpt:
                    generator.load_state_dict(ckpt["generator_state"], strict=False)
                elif "model_state_dict" in ckpt:
                    generator.load_state_dict(ckpt["model_state_dict"], strict=False)
                else:
                    generator.load_state_dict(ckpt, strict=False)
                meta["epoch"] = ckpt.get("epoch", None)
                meta["val_psnr"] = ckpt.get("val_psnr", None)
                meta["val_ssim"] = ckpt.get("val_ssim", None)
            else:
                generator.load_state_dict(ckpt, strict=False)
            loaded = True
        except Exception as e:
            st.warning(f"Error loading checkpoint: {e}")

    generator.eval()
    return generator, device, loaded, meta


def main():
    st.title("LumaCam: Edge-Native Adaptive Low-Light Computational Photography")
    st.markdown(
        "**Team 21 (SCTCE)** | Capstone Project Guided by **Asst. Prof. Arya Sreedhar**  \n"
        "*Exposure-Conditioned Attention U-Net GAN with FiLM Conditioning & Edge-Native Color Science.*"
    )

    # Sidebar Controls
    st.sidebar.header("Camera & Model Controls")

    ckpt_default = "checkpoints/lumacam_best.pt"
    ckpt_path = st.sidebar.text_input("Checkpoint Path", value=ckpt_default)

    with st.spinner("Initializing neural engine..."):
        generator, device, is_loaded, meta = load_lumacam_model(ckpt_path)

    status_icon = "[ACTIVE]" if is_loaded else "[PENDING]"
    st.sidebar.markdown(f"**Engine Status**: {status_icon} Loaded on `{device}`")
    if device.type == "cuda":
        st.sidebar.caption(f"Hardware: {torch.cuda.get_device_name(0)}")
    if meta.get("epoch") is not None:
        st.sidebar.caption(f"Checkpoint Epoch: {meta['epoch']} | Val PSNR: {meta.get('val_psnr', 0.0):.2f} dB")

    st.sidebar.markdown("---")
    st.sidebar.subheader("User Exposure Slider (α)")
    alpha_val = st.sidebar.slider(
        "Exposure Amplification (α)",
        min_value=10.0,
        max_value=300.0,
        value=100.0,
        step=5.0,
        help="Simulates the tactile GCam exposure slider modulating FiLM layers at the U-Net bottleneck. Low-light amplification in SID and Night Sight ranges from 50x to 300x.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Post-Inference Color Science & ISP")
    enable_color_science = st.sidebar.checkbox(
        "Enable Color Science (AWB & Saturation)",
        value=True,
        help="Gray-World Auto White Balance in linear light + hue-preserving saturation.",
    )
    awb_strength = st.sidebar.slider("AWB Strength", 0.0, 1.0, 0.80, 0.05) if enable_color_science else 0.0
    sat_factor = st.sidebar.slider("Saturation Multiplier", 0.8, 1.5, 1.10, 0.05) if enable_color_science else 1.0

    enable_isp_sharpness = st.sidebar.checkbox(
        "Enable Dynamic Range & Edge Sharpening",
        value=True,
        help="Applies percentile contrast normalization to remove foggy black elevation, plus edge-preserving unsharp masking to synthesize crisp textures.",
    )
    sharpen_amount = st.sidebar.slider("Sharpening Strength", 0.0, 2.5, 1.6, 0.1) if enable_isp_sharpness else 0.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("Input Source")
    input_mode = st.sidebar.radio(
        "Select Image Source",
        options=[
            "Preloaded Low-Light Photo (Indoor Scene)",
            "Upload Custom Image / Crop",
            "Synthetic Poisson-Gaussian Sensor Simulation",
        ],
    )

    # Initialize sample
    patch_size = 512
    if input_mode == "Preloaded Low-Light Photo (Indoor Scene)":
        preset_path = REPO_ROOT / "Report" / "Assets" / "ground_truth.png"
        if preset_path.exists():
            pil_img = Image.open(preset_path).convert("RGB").resize((patch_size, patch_size))
            img_np = np.array(pil_img, dtype=np.float32) / 255.0
        else:
            img_np = np.ones((patch_size, patch_size, 3), dtype=np.float32) * 0.5
        gt_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # Synthesize underexposed Bayer pattern with physical Poisson-Gaussian CMOS noise
        r = img_np[0::2, 0::2, 0]
        g1 = img_np[0::2, 1::2, 1]
        g2 = img_np[1::2, 0::2, 1]
        b = img_np[1::2, 1::2, 2]
        packed = np.stack([r, g1, b, g2], axis=0)  # (4, H/2, W/2)
        dark_packed = packed / max(alpha_val, 1.0)
        shot_noise = np.random.normal(0, np.sqrt(np.maximum(dark_packed, 0.0) / 40.0 + 1e-5)) * 0.02
        read_noise = np.random.normal(0, 0.015, dark_packed.shape)
        dark_packed = np.clip(dark_packed + shot_noise + read_noise, 0.0, 1.0)
        bayer_tensor = torch.from_numpy(dark_packed * alpha_val).float()

    elif input_mode == "Upload Custom Image / Crop":
        uploaded_file = st.sidebar.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB").resize((patch_size, patch_size))
            img_np = np.array(pil_img, dtype=np.float32) / 255.0
            gt_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

            # Synthesize underexposed Bayer pattern with physical Poisson-Gaussian CMOS noise
            r = img_np[0::2, 0::2, 0]
            g1 = img_np[0::2, 1::2, 1]
            g2 = img_np[1::2, 0::2, 1]
            b = img_np[1::2, 1::2, 2]
            packed = np.stack([r, g1, b, g2], axis=0)  # (4, H/2, W/2)
            dark_packed = packed / max(alpha_val, 1.0)
            shot_noise = np.random.normal(0, np.sqrt(np.maximum(dark_packed, 0.0) / 40.0 + 1e-5)) * 0.02
            read_noise = np.random.normal(0, 0.015, dark_packed.shape)
            dark_packed = np.clip(dark_packed + shot_noise + read_noise, 0.0, 1.0)
            bayer_tensor = torch.from_numpy(dark_packed * alpha_val).float()
        else:
            st.info("Upload an image or switch to Preloaded Low-Light Photo.")
            dataset = DummySIDDataset(num_samples=10, patch_size=patch_size, seed=42)
            bayer_tensor, gt_tensor, _ = dataset[0]
    else:
        seed = st.sidebar.number_input("Random Scene Seed", value=42, step=1)
        dataset = DummySIDDataset(num_samples=10, patch_size=patch_size, seed=int(seed))
        bayer_tensor, gt_tensor, _ = dataset[0]

    # Previews
    # Resolution scaling factors
    b_h, b_w = bayer_tensor.shape[1], bayer_tensor.shape[2]
    scale_y = patch_size // b_h
    scale_x = patch_size // b_w

    # 1. Dark sensor preview (un-amplified physical raw Bayer frame arriving from CMOS sensor)
    dark_bayer = (bayer_tensor / max(alpha_val, 1.0)).numpy()
    r_d = dark_bayer[0]
    g_d = 0.5 * (dark_bayer[1] + dark_bayer[3])
    b_d = dark_bayer[2]
    raw_dark_half = np.stack([r_d, g_d, b_d], axis=-1)
    raw_dark = np.repeat(np.repeat(raw_dark_half, scale_y, axis=0), scale_x, axis=1)
    raw_dark = np.clip(raw_dark, 0.0, 1.0)

    # 2. Traditional Linear ISO Gain (shows authentic sensor noise blow-up and chromatic speckling)
    r_lin = bayer_tensor[0].numpy()
    g_lin = 0.5 * (bayer_tensor[1].numpy() + bayer_tensor[3].numpy())
    b_lin = bayer_tensor[2].numpy()
    linear_half = np.stack([r_lin, g_lin, b_lin], axis=-1)
    linear_boost_3ch = np.repeat(np.repeat(linear_half, scale_y, axis=0), scale_x, axis=1)
    linear_boost_3ch = np.clip(linear_boost_3ch, 0.0, 1.0)

    # 3. LumaCam Neural Model Forward Pass
    inp_bayer = bayer_tensor.unsqueeze(0).to(device)
    inp_alpha = torch.tensor([[alpha_val]], device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        enhanced_t, attn_maps = generator(inp_bayer, inp_alpha, return_attention_maps=True)
        enhanced_t = torch.clamp(enhanced_t, 0.0, 1.0)

        # Apply post-inference color science
        if enable_color_science:
            final_t = apply_color_pipeline(
                enhanced_t,
                awb_strength=awb_strength,
                saturation=sat_factor,
                input_is_linear=False,
            )
        else:
            final_t = enhanced_t

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    enhanced_raw_np = enhanced_t[0].permute(1, 2, 0).cpu().numpy()
    # Normalize dynamic range on preview 3 so it's not foggy
    p2_raw, p98_raw = np.percentile(enhanced_raw_np, (2, 98))
    if p98_raw > p2_raw + 1e-4:
        enhanced_raw_np = np.clip((enhanced_raw_np - p2_raw) / (p98_raw - p2_raw), 0.0, 1.0)

    final_np = final_t[0].permute(1, 2, 0).cpu().numpy()
    if enable_isp_sharpness:
        # Dynamic range stretch to restore deep blacks and clean highlights
        p1, p99 = np.percentile(final_np, (1, 99))
        if p99 > p1 + 1e-4:
            final_np = np.clip((final_np - p1) / (p99 - p1), 0.0, 1.0)

        # ISP Edge Sharpening (Unsharp Mask for crisp facial and architectural details)
        if sharpen_amount > 0.0:
            final_pil = Image.fromarray((final_np * 255.0).astype(np.uint8))
            sharp_pil = final_pil.filter(
                ImageFilter.UnsharpMask(radius=1.2, percent=int(sharpen_amount * 100), threshold=2)
            )
            final_np = np.array(sharp_pil, dtype=np.float32) / 255.0

    gt_np = gt_tensor.permute(1, 2, 0).numpy()

    # Metrics computation
    mse_linear = float(np.mean((linear_boost_3ch - gt_np) ** 2))
    psnr_linear = 20.0 * np.log10(1.0 / np.sqrt(mse_linear)) if mse_linear > 1e-10 else 100.0

    mse_luma = float(np.mean((final_np - gt_np) ** 2))
    psnr_luma = 20.0 * np.log10(1.0 / np.sqrt(mse_luma)) if mse_luma > 1e-10 else 100.0

    # Display Top Metrics Cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Exposure Gain (α)", f"{alpha_val:.1f}x")
    with col_m2:
        st.metric("Inference Latency", f"{latency_ms:.2f} ms", delta=f"{1000.0/latency_ms:.1f} FPS")
    with col_m3:
        st.metric("Linear Boost PSNR", f"{psnr_linear:.2f} dB")
    with col_m4:
        st.metric("LumaCam Enhanced PSNR", f"{psnr_luma:.2f} dB", delta=f"{psnr_luma - psnr_linear:+.2f} dB")

    st.markdown("---")
    st.subheader("Computational Photography Pipeline Comparison")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(raw_dark, caption=f"1. Physical Sensor Bayer (α = {alpha_val:.1f}x)", clamp=True, use_container_width=True)
    with col2:
        st.image(linear_boost_3ch, caption="2. Traditional Linear ISO Gain (Noise Blow-up)", clamp=True, use_container_width=True)
    with col3:
        st.image(enhanced_raw_np, caption="3. Attention U-Net GAN Output (Pre-Color)", clamp=True, use_container_width=True)
    with col4:
        st.image(final_np, caption=f"4. LumaCam Final (+ Color Science: {psnr_luma:.2f} dB)", clamp=True, use_container_width=True)

    # Attention Gate Explainability Section
    st.markdown("---")
    st.subheader("Explainable AI: Spatial Attention Gate Noise Suppression")
    st.markdown(
        "Spatial Attention Gates on skip connections compute soft attention maps $\\alpha_{att} \\in [0, 1]$ "
        "conditioned on coarser gating features from the decoder. This filters out severe Poisson photon noise in dark areas "
        "while transmitting sharp edge boundaries."
    )

    if attn_maps is not None:
        ag_cols = st.columns(4)
        for i, (a_map, c) in enumerate(zip(attn_maps, ag_cols), start=1):
            att_img = a_map[0, 0].cpu().numpy()
            with c:
                st.image(
                    att_img,
                    caption=f"Attention Gate {i} ({att_img.shape[0]}×{att_img.shape[1]})",
                    clamp=True,
                    use_container_width=True,
                )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Presentation Tip**: Slide exposure α to demonstrate how the FiLM layer adaptively balances highlights and shadows in a single forward pass without re-demosaicing!"
    )


if __name__ == "__main__":
    main()
