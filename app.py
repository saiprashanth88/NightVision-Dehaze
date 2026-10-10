import os
import io
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import NightDehazeNet
from inference import enhance_tiled, safe_quality_guard, laplacian_sharpness

st.set_page_config(
    page_title="NightVision-Dehaze | Nighttime Visibility Analysis",
    page_icon="🌙",
    layout="wide",
)

CHECKPOINT = "checkpoints/acdc_v2_best.pth"

GEMINI_STAGE_PROMPTS = {
    "1 · Visibility preprocessing": (
        "Enhance this nighttime road photo as a preprocessing demonstration: gently reduce sensor noise, "
        "balance exposure, preserve natural colors and keep the exact camera view and scene geometry."
    ),
    "2 · Haze reduction": (
        "Edit this nighttime road photo to reduce visible atmospheric fog and haze while retaining realistic "
        "nighttime lighting, the same road layout, lane markings, vehicles, signs and buildings."
    ),
    "3 · Low-light recovery": (
        "Improve visibility in underexposed regions of this nighttime road photo. Lift dark road and roadside "
        "details naturally, preserve headlight highlights, and do not add or move objects."
    ),
    "4 · Glare control": (
        "Reduce excessive headlight bloom and bright glare in this nighttime road photo while preserving "
        "realistic light sources, vehicle positions, lane markings, and the original scene composition."
    ),
    "5 · Detail clarity": (
        "Improve perceived local contrast and edge clarity in this nighttime road photo without creating "
        "halos, oversharpening, changing object identities, or inventing hidden details."
    ),
    "6 · Demo final composite": (
        "Create a natural-looking visibility-enhanced version of this exact nighttime road photograph. "
        "Balance haze reduction, low-light visibility, highlight control and fine detail. Preserve camera "
        "viewpoint, road geometry, lane markings, vehicles, signs and buildings. Do not add, remove, duplicate "
        "or relocate objects. This is a visual demo edit, not scientific ground-truth restoration."
    ),
}


def get_gemini_api_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def generate_gemini_stage_image(image_bytes, prompt, api_key):
    """Generate one optional Gemini image-edit preview; never changes the trained-model pipeline."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    response = client.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=[
            prompt + " Return an edited image only. Preserve the source aspect ratio where possible.",
            source_image,
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    for part in response.parts:
        if getattr(part, "inline_data", None) is not None:
            raw = part.inline_data.data
            if raw:
                return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    raise RuntimeError("Gemini returned no image. Check model access, API quota, and billing.")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CHECKPOINT):
        return None, device, 256, "Checkpoint not found"
    try:
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model = NightDehazeNet(base=ckpt.get("base", 24)).to(device).float()
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, device, int(ckpt.get("size", 256)), "Loaded"
    except Exception as exc:
        return None, device, 256, f"Could not load checkpoint: {exc}"


def to_float(rgb):
    return np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)


def to_uint8(rgb01):
    return np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)


def preprocess_image(rgb):
    """Light, conservative denoising and normalization for visualization."""
    denoised = cv2.fastNlMeansDenoisingColored(rgb, None, 3, 3, 7, 21)
    # Keep this stage conservative: don't aggressively brighten the source.
    return denoised


def condition_analysis(rgb):
    """Heuristic visual indicators, not a trained condition classifier."""
    f = to_float(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    mean_luma = float(np.mean(gray_f))
    contrast = float(np.std(gray_f))
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    bright_fraction = float(np.mean(gray >= 245))
    dark_fraction = float(np.mean(gray <= 45))

    # A rough fog/haze cue: low contrast with limited color saturation.
    fog_score = float(np.clip((0.22 - contrast) / 0.18 * 0.65 + (0.38 - saturation) * 0.35, 0, 1))
    low_light_score = float(np.clip((0.42 - mean_luma) / 0.38, 0, 1))
    glare_score = float(np.clip(bright_fraction * 18.0 + max(0.0, mean_luma - 0.55) * 0.35, 0, 1))

    # Simple explanatory map: blue = darker pixels, green = mid-tones,
    # red = very bright pixels. It is a visualization, not semantic segmentation.
    dark = np.clip((0.45 - gray_f) / 0.45, 0, 1)
    bright = np.clip((gray_f - 0.78) / 0.22, 0, 1)
    mid = np.clip(1.0 - dark - bright, 0, 1)
    heat = np.stack([bright, mid * 0.75, dark], axis=-1)
    heat = np.clip(heat * 255, 0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(rgb, 0.48, heat, 0.52, 0)

    def level(score):
        if score >= 0.66:
            return "High"
        if score >= 0.34:
            return "Moderate"
        return "Low"

    return {
        "fog_score": fog_score,
        "low_light_score": low_light_score,
        "glare_score": glare_score,
        "brightness": mean_luma,
        "contrast": contrast,
        "saturation": saturation,
        "sharpness": laplacian_sharpness(f),
        "bright_fraction": bright_fraction,
        "dark_fraction": dark_fraction,
        "fog_label": level(fog_score),
        "low_light_label": "Detected" if low_light_score >= 0.34 else "Low",
        "glare_label": "Detected" if glare_score >= 0.34 else "Low",
        "overlay": overlay,
    }


def classic_enhancement(rgb, strength=0.45):
    """Fallback when the trained checkpoint is unavailable; not a learned model."""
    f = to_float(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

    # Gentle gamma lift for dark scenes, limited to avoid washing out highlights.
    gray_mean = float(np.mean(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))) / 255.0
    gamma = 0.82 if gray_mean < 0.38 else 0.94
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    lifted = cv2.LUT(contrast, lut)

    blur = cv2.GaussianBlur(lifted, (0, 0), 1.0)
    sharp = cv2.addWeighted(lifted, 1.12, blur, -0.12, 0)
    alpha = float(np.clip(strength, 0, 1))
    out = cv2.addWeighted(rgb, 1.0 - alpha, sharp, alpha, 0)
    return to_float(out)


def postprocess_image(rgb01, strength=0.45):
    img = to_uint8(rgb01)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharp = cv2.addWeighted(contrast, 1.10, blur, -0.10, 0)
    alpha = float(np.clip(strength, 0, 1))
    return to_float(cv2.addWeighted(img, 1.0 - alpha, sharp, alpha, 0))


def image_metrics(rgb01):
    gray = cv2.cvtColor(to_uint8(rgb01), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return {
        "Sharpness (Laplacian variance)": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "Contrast (gray std.)": float(np.std(gray)),
        "Mean brightness": float(np.mean(gray)),
    }


def reference_metrics(candidate, reference):
    if candidate.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (candidate.shape[1], candidate.shape[0]), interpolation=cv2.INTER_AREA)
    cand = to_float(candidate)
    ref = to_float(reference)
    return (
        float(peak_signal_noise_ratio(ref, cand, data_range=1.0)),
        float(structural_similarity(ref, cand, channel_axis=2, data_range=1.0)),
    )


def png_bytes(rgb01):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(to_uint8(rgb01), cv2.COLOR_RGB2BGR))
    return encoded.tobytes() if ok else b""


def show_metric_graphs(stages):
    """Display colorful, aligned PSNR and SSIM bars for the same six pipeline stages."""
    st.markdown("## 📈 PSNR Evaluation")
    st.caption("Each bar uses the original uploaded image as the comparison reference. These are input-similarity scores, not ground-truth restoration scores.")

    labels = list(stages.keys())
    psnr_values = []
    ssim_values = []
    for name, stage_img in stages.items():
        stage_u8 = to_uint8(stage_img)
        reference_u8 = to_uint8(stages["Input"])
        p, s = reference_metrics(stage_u8, reference_u8)
        psnr_values.append(p)
        ssim_values.append(s)

    # The input compared with itself has infinite PSNR. Cap only its bar height
    # for readable plotting and label it explicitly as infinity.
    finite_psnr = [v for v in psnr_values if np.isfinite(v)]
    psnr_cap = max(40.0, (max(finite_psnr) + 5.0) if finite_psnr else 40.0)
    psnr_plot = [psnr_cap if not np.isfinite(v) else v for v in psnr_values]
    psnr_labels = ["∞" if not np.isfinite(v) else f"{v:.1f}" for v in psnr_values]

    def draw_stage_chart(values, title, ylabel, color, value_labels, ylim=None):
        fig, ax = plt.subplots(figsize=(12.8, 4.5))
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=color, width=0.68, edgecolor="white", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color="#17365D", pad=14)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.set_axisbelow(True)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, max(values) * 1.18 if values else 1)
        for bar, label in zip(bars, value_labels):
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#17365D",
            )
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#FBFCFE")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    draw_stage_chart(
        psnr_plot,
        "PSNR across processing stages",
        "PSNR (dB) — similarity to original input",
        ["#3498DB", "#5DADE2", "#8E7CC3", "#17A589", "#F39C12", "#27AE60"],
        psnr_labels,
    )

    st.markdown("## 📈 SSIM Evaluation")
    st.caption("SSIM ranges from 0 to 1. A score closer to 1 means the stage image is more structurally similar to the original input.")
    draw_stage_chart(
        ssim_values,
        "SSIM across processing stages",
        "SSIM (0–1)",
        ["#3498DB", "#5DADE2", "#8E7CC3", "#17A589", "#F39C12", "#27AE60"],
        [f"{v:.2f}" for v in ssim_values],
        ylim=(0, 1.12),
    )
    st.info(
        "The Input bars are the reference baseline (SSIM = 1 and PSNR = ∞). The PSNR bar for Input is visually capped so the graph remains readable. "
        "To evaluate true dehazing quality, compare every stage against a corresponding clean ground-truth image."
    )

    # Additional input-versus-final image property comparisons.
    st.markdown("## 🎨 Input vs. Final Output — Image Property Comparisons")
    input_rgb = to_uint8(stages["Input"])
    final_rgb = to_uint8(stages["Final Output"])
    input_gray = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    final_gray = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    input_hsv = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2HSV)
    final_hsv = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2HSV)

    comparisons = [
        ("Brightness comparison", "Mean grayscale brightness (0–1)",
         [float(input_gray.mean()), float(final_gray.mean())], ["#3498DB", "#2ECC71"], (0, 1)),
        ("Contrast comparison", "Grayscale standard deviation (0–1)",
         [float(input_gray.std()), float(final_gray.std())], ["#8E7CC3", "#F39C12"], None),
        ("Sharpness comparison", "Laplacian variance",
         [float(cv2.Laplacian(input_gray, cv2.CV_32F).var()),
          float(cv2.Laplacian(final_gray, cv2.CV_32F).var())], ["#3498DB", "#2ECC71"], None),
        ("Color saturation comparison", "Mean HSV saturation (0–255)",
         [float(input_hsv[:, :, 1].mean()), float(final_hsv[:, :, 1].mean())], ["#E67E22", "#16A085"], (0, 255)),
    ]

    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)
    for idx, (chart_title, ylabel, values, colors, limits) in enumerate(comparisons):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        bars = ax.bar(["Input", "Final Output"], values, color=colors, width=0.58,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(chart_title, fontsize=14, fontweight="bold", color="#17365D", pad=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        if limits is not None:
            ax.set_ylim(*limits)
        else:
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax.bar_label(bars, fmt="%.3f" if idx < 2 else ("%.2f" if idx == 2 else "%.1f"),
                     padding=4, fontsize=9, fontweight="bold")
        ax.set_facecolor("#FBFCFE")
        fig.patch.set_facecolor("white")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        target = [row1_left, row1_right, row2_left, row2_right][idx]
        with target:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)


st.title("🌙 Nighttime Visibility Analysis")
st.caption("Night Time Defogging Vision Network with Smart Glare Suppression")
st.markdown(
    "Upload one or more road images to see the processing pipeline, heuristic fog/low-light/glare indicators, "
    "final enhancement, and image-quality graphs."
)

model, device, tile_size, model_status = load_model()
with st.sidebar:
    st.header("Analysis settings")
    st.write(f"**Device:** {'GPU' if device.type == 'cuda' else 'CPU'}")
    st.write(f"**V2 model:** {model_status}")
    if model is None:
        st.warning("The trained checkpoint is not available. The app will use a classical OpenCV fallback for the demo.")
        st.caption("The fallback is not your trained CNN + Transformer model.")
    strength = st.slider("Post-processing strength", 0.0, 1.0, 0.45, 0.05)
    show_intermediate = st.checkbox("Show all processing stages", value=True)
    st.markdown("---")
    st.subheader("Temporary Gemini demo layer")
    use_gemini_demo = st.checkbox("Enable Gemini stage renders", value=False)
    gemini_api_key = get_gemini_api_key()
    if use_gemini_demo and not gemini_api_key:
        st.warning("Set GEMINI_API_KEY in your environment or Streamlit secrets to enable Gemini renders.")
    st.caption("Optional image-generation demo only. The trained model and its outputs remain unchanged.")
    if use_gemini_demo:
        st.warning("Gemini image-generation models may be paid-only; check your Google AI Studio/API billing before generating. Only the final composite uses Gemini; all other pipeline stages continue to use the existing processing/model.")

uploads = st.file_uploader(
    "Upload nighttime / foggy / rainy / snowy road images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="road_images",
)

if not uploads:
    st.info("Upload one or more images to generate the full analysis screen. You can test the dashboard without training; the OpenCV fallback will be used if the V2 checkpoint is missing.")
    st.markdown("#### Dashboard preview")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Fog detection", "Awaiting image")
    p2.metric("Low-light", "Awaiting image")
    p3.metric("Glare", "Awaiting image")
    p4.metric("Image clarity", "Awaiting image")
else:
    for index, uploaded in enumerate(uploads):
        try:
            original_rgb = np.array(Image.open(uploaded).convert("RGB"))
        except Exception as exc:
            st.error(f"Could not read {uploaded.name}: {exc}")
            continue

        original01 = to_float(original_rgb)
        preprocessed = preprocess_image(original_rgb)
        condition = condition_analysis(original_rgb)

        with st.expander(f"Image {index + 1}: {uploaded.name}", expanded=(index == 0)):
            if use_gemini_demo and gemini_api_key:
                st.markdown("### Optional Gemini final-image demonstration")
                st.caption("Only the final composite is generated by Gemini for presentation. All earlier pipeline stages and their graphs remain from the existing trained model/processing pipeline.")
                stage_key = __import__("hashlib").sha256(uploaded.getvalue()).hexdigest()
                cache = st.session_state.setdefault("gemini_stage_demo_cache", {})
                cache.setdefault(stage_key, {})
                final_stage_name = "6 · Demo final composite"
                if st.button(
                    "Generate Gemini final composite",
                    key=f"generate_gemini_final_{index}_{stage_key[:8]}",
                ):
                    if final_stage_name not in cache[stage_key]:
                        with st.spinner("Generating Gemini final composite..."):
                            try:
                                cache[stage_key][final_stage_name] = {
                                    "image": generate_gemini_stage_image(
                                        uploaded.getvalue(),
                                        GEMINI_STAGE_PROMPTS[final_stage_name],
                                        gemini_api_key,
                                    ),
                                    "error": None,
                                }
                            except Exception as exc:
                                cache[stage_key][final_stage_name] = {
                                    "image": None,
                                    "error": str(exc),
                                }

                item = cache[stage_key].get(final_stage_name)
                if item:
                    st.markdown("**Gemini final composite**")
                    if item["image"] is not None:
                        source_col, result_col = st.columns(2)
                        source_col.image(original_rgb, caption="Original input", width="stretch")
                        result_col.image(item["image"], caption="Gemini-enhanced final demo image", width="stretch")
                        png = cv2.imencode(
                            ".png", cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
                        )[1].tobytes()
                        st.download_button(
                            "Download Gemini final composite",
                            data=png,
                            file_name=f"{os.path.splitext(uploaded.name)[0]}_gemini_final_composite.png",
                            mime="image/png",
                            key=f"gemini_final_download_{index}_{stage_key[:8]}",
                        )
                    elif item["error"]:
                        st.error(f"Gemini final composite failed: {item['error']}")
            st.subheader("Nighttime Visibility Analysis")
            st.caption(f"Image size: {original_rgb.shape[1]} × {original_rgb.shape[0]} px")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fog / haze cue", condition["fog_label"], f'{condition["fog_score"] * 100:.0f}% heuristic score')
            m2.metric("Low-light cue", condition["low_light_label"], f'{condition["low_light_score"] * 100:.0f}% heuristic score')
            m3.metric("Bright glare cue", condition["glare_label"], f'{condition["glare_score"] * 100:.0f}% heuristic score')
            m4.metric("Image clarity", f'{condition["sharpness"]:.1f}', "Laplacian sharpness")

            with st.spinner(f"Processing {uploaded.name}..."):
                pre01 = to_float(preprocessed)
                if model is not None:
                    try:
                        raw_model = enhance_tiled(model, preprocessed, device, tile=tile_size, overlap=min(64, tile_size // 4))
                        model_out, original_sharp, model_sharp, mean_change, guarded = safe_quality_guard(pre01, raw_model)
                        method_label = "Trained V2 CNN + Channel Attention + Transformer"
                    except Exception as exc:
                        st.warning(f"V2 inference failed for this image; using OpenCV fallback. Details: {exc}")
                        model_out = classic_enhancement(preprocessed, strength)
                        original_sharp = laplacian_sharpness(pre01)
                        model_sharp = laplacian_sharpness(model_out)
                        mean_change = float(np.mean(np.abs(model_out - pre01)))
                        guarded = False
                        method_label = "OpenCV fallback (V2 inference failed)"
                else:
                    model_out = classic_enhancement(preprocessed, strength)
                    original_sharp = laplacian_sharpness(pre01)
                    model_sharp = laplacian_sharpness(model_out)
                    mean_change = float(np.mean(np.abs(model_out - pre01)))
                    guarded = False
                    method_label = "OpenCV classical enhancement fallback"

                # Expose six clearly named, reproducible visual stages.
                condition_visual = to_float(condition["overlay"])
                feature_fusion = np.clip(0.75 * model_out + 0.25 * pre01, 0.0, 1.0)
                final = postprocess_image(feature_fusion, strength)
                stages = {
                    "Input": original01,
                    "Preprocessing": pre01,
                    "Condition Analysis": condition_visual,
                    "CNN + ViT": model_out,
                    "Feature Fusion": feature_fusion,
                    "Final Output": final,
                }

            st.markdown(f"**Processing method:** {method_label}")
            if guarded:
                st.warning("The quality guard blended the model output toward the preprocessed input because a large change or sharpness drop was detected.")
            elif model is None:
                st.warning("The trained checkpoint was not loaded; this result is produced by classical image processing, not by the trained neural network.")

            if show_intermediate:
                columns = st.columns(4)
                columns[0].image(original_rgb, caption="1. Input image", width="stretch")
                columns[1].image(preprocessed, caption="2. Preprocessing (denoise)", width="stretch")
                columns[2].image(condition["overlay"], caption="3. Condition-analysis visualization", width="stretch")
                columns[3].image(to_uint8(model_out), caption="4. Neural / fallback enhancement stage", width="stretch")
                st.markdown("#### Final enhanced output")
                left, right = st.columns([1.5, 1])
                with left:
                    st.image(to_uint8(final), caption="Final output (contrast and sharpening refinement)", width="stretch")
                with right:
                    st.markdown("**Condition indicators**")
                    st.progress(condition["fog_score"], text=f'Fog / haze cue: {condition["fog_score"] * 100:.0f}%')
                    st.progress(condition["low_light_score"], text=f'Low-light cue: {condition["low_light_score"] * 100:.0f}%')
                    st.progress(condition["glare_score"], text=f'Bright-glare cue: {condition["glare_score"] * 100:.0f}%')
                    st.caption("These indicators are simple image-statistic heuristics, not a validated detection model.")
            else:
                left, right = st.columns(2)
                left.image(original_rgb, caption="Original input", width="stretch")
                right.image(to_uint8(final), caption="Final enhanced output", width="stretch")

            sharp_a, sharp_b, delta = st.columns(3)
            sharp_a.metric("Input sharpness", f"{original_sharp:.2f}")
            sharp_b.metric("Enhancement-stage sharpness", f"{model_sharp:.2f}")
            delta.metric("Mean pixel change", f"{mean_change:.4f}")

            show_metric_graphs(stages)

            st.download_button(
                "Download final enhanced image",
                data=png_bytes(final),
                file_name=f"{os.path.splitext(uploaded.name)[0]}_nightvision_enhanced.png",
                mime="image/png",
                key=f"download_{index}_{uploaded.name}",
            )

st.markdown("---")
st.caption(
    "Research/demo prototype. Brightness, fog and glare indicators are heuristic estimates. "
    "Estimated visibility in metres is intentionally not displayed because this app has no calibrated distance-estimation model."
)
