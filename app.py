# ─────────────────────────────────────────────────────────
# app.py — MammoAI · Breast Cancer Detection (single-page dashboard)
# Sidebar (about + Q&A chatbot) + main panel (upload, Grad-CAM, results)
# ─────────────────────────────────────────────────────────
import torch
from PIL import Image

import streamlit as st

from core.config import DENSITY_MAP, VIEW_MAP, MODEL_PATH, SCREENING_THRESHOLD
from core.chatbot import handle_chat

st.set_page_config(
    page_title="MammoAI — Breast Cancer Detection",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Theme — deep navy / indigo-cyan accent, "clinical AI" palette
# ─────────────────────────────────────────────────────────
DARK_THEME_CSS = """
<style>
:root {
    --bg:        #0a0f1e;
    --panel:     #131a2e;
    --panel-2:   #182238;
    --border:    #25304a;
    --text:      #e6ebf5;
    --text-dim:  #8b97b0;
    --accent:    #6366f1;
    --accent-2:  #22d3ee;
    --good:      #34d399;
    --bad:       #fb7185;
}

.stApp, .main {
    background: radial-gradient(circle at 15% -10%, #16213d 0%, var(--bg) 45%) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #111729 0%, #0d1322 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text); }
section[data-testid="stSidebar"] h3 {
    background: linear-gradient(90deg, var(--accent-2), var(--accent));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    letter-spacing: .2px;
}

/* ── Typography ── */
h1 {
    background: linear-gradient(90deg, #ffffff 0%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: .3px;
}
h2 { color: var(--text) !important; font-weight: 700 !important; letter-spacing: .2px; }
h3 { color: var(--text) !important; font-weight: 650 !important; }
p, li, span, label { color: var(--text-dim); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-dim) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    transition: border-color .2s ease;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-2); }
[data-testid="stFileUploader"] section { background: transparent; }
[data-testid="stFileUploader"] button {
    background: var(--panel-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
}
[data-testid="stFileUploader"] button:hover { border-color: var(--accent-2); color: var(--accent-2); }

/* ── Selects ── */
[data-baseweb="select"] > div {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
[data-baseweb="select"] * { color: var(--text) !important; }

/* ── Images / captions ── */
.stImage img {
    border-radius: 10px;
    border: 1px solid var(--border);
    box-shadow: 0 6px 24px rgba(0,0,0,.35);
}
.stImage > div > div > div > p,
[data-testid="stImageCaption"] {
    color: var(--text-dim) !important;
    font-size: 12px;
    letter-spacing: .4px;
    text-transform: uppercase;
    text-align: center;
    margin-top: 6px;
}

/* ── Buttons (sidebar quick-questions etc.) ── */
.stButton > button {
    background: var(--panel-2);
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 8px;
    text-align: left;
    font-size: 13px;
    padding: 9px 12px;
    transition: all .15s ease;
}
.stButton > button:hover {
    border-color: var(--accent-2);
    color: var(--text);
    box-shadow: 0 0 0 1px var(--accent-2) inset;
}

/* ── Primary action button (Run Classification) ── */
div[data-testid="stSidebar"] .stButton > button[kind="primary"],
.main .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
    color: #06121f;
    border: none;
    text-align: center;
    font-weight: 700;
    letter-spacing: .3px;
    box-shadow: 0 8px 24px rgba(99,102,241,.35);
}
.main .stButton > button[kind="primary"]:hover {
    box-shadow: 0 10px 30px rgba(34,211,238,.45);
    transform: translateY(-1px);
}
.main .stButton > button[kind="primary"]:disabled {
    background: var(--panel-2);
    color: var(--text-dim);
    box-shadow: none;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 4px 6px;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }

hr { border-color: var(--border); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-dim);
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-2);
    border-bottom: 2px solid var(--accent-2);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* ── Alerts (info/warning/error) ── */
[data-testid="stAlert"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
</style>
"""
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Model loading (cached across reruns)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    from core.model import load_model
    return load_model(MODEL_PATH, device="cpu")


try:
    model, _trained_threshold = get_model()
    # Override the checkpoint's F1-optimal threshold with a lower, sensitivity-
    # leaning one (see core/config.SCREENING_THRESHOLD) — for a screening tool
    # we'd rather flag a normal case for review than miss an abnormal one.
    threshold = SCREENING_THRESHOLD
    model_load_error = None
except Exception as exc:
    model, threshold = None, SCREENING_THRESHOLD
    model_load_error = str(exc)


# ─────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("result", None)
st.session_state.setdefault("result_ready", False)
st.session_state.setdefault("chat_prefill", None)


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🩺 About This Detection")
    st.markdown("**How it works:**")
    st.markdown(
        "1. Upload a mammogram image\n"
        "2. AI analyzes key biomarkers\n"
        "3. Get classification + confidence\n"
        "4. Chat with the medical AI"
    )

    st.divider()
    st.markdown("### Breast Cancer Q&A Chatbot")

    tab_pred, tab_info = st.tabs(["Prediction Q&A", "General Info"])

    with tab_pred:
        st.markdown("**Ask About Your Results**")

        quick_questions = [
            "Does this patient have breast cancer?",
            "Why did the model predict this result?",
            "What does the Grad-CAM highlight?",
            "Should I seek further screening?",
        ]
        for q in quick_questions:
            if st.button(q, key=f"quick_{q}", use_container_width=True):
                st.session_state.chat_prefill = q

        for msg in st.session_state.messages:
            avatar = "🔴" if msg["role"] == "user" else "🟡"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        prefill = st.session_state.pop("chat_prefill", None)
        prompt = st.chat_input("Ask about your results…")
        if prefill and not prompt:
            prompt = prefill
        if prompt:
            handle_chat(prompt, st.session_state.result)

    with tab_info:
        st.markdown(
            "**VinDr-Mammo Dataset**\n"
            "- 20,137 mammograms\n"
            "- BI-RADS 1–2 → Normal\n"
            "- BI-RADS 3+ → Abnormal\n\n"
            "**Model: EfficientNet-B3 NS**\n"
            "- Dual-stream architecture\n"
            "- Clinical metadata fusion\n"
            "- Grad-CAM explainability"
        )


# ══════════════════════════════════════════════════════════
#  MAIN PANEL
# ══════════════════════════════════════════════════════════
st.markdown("# 🩺 MammoAI — Breast Cancer Detection")
st.caption("EfficientNet-B3 NoisyStudent · Dual-Stream · VinDr-Mammo · Thesis Project")

if model_load_error:
    st.error(f"⚠️ Could not load the model weights: {model_load_error}")

# ── 1. File uploader ────────────────────────────────────
st.markdown(
    """
    <div style='background:linear-gradient(135deg, #131a2e 0%, #182238 100%);
                border:1px dashed #25304a; border-radius:14px; padding:22px 24px;
                margin-bottom:8px; display:flex; align-items:center; gap:16px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.03);'>
      <span style='font-size:28px; filter: drop-shadow(0 0 10px rgba(34,211,238,.35));'>☁️</span>
      <div>
        <div style='color:#e6ebf5; font-weight:600; font-size:15px;'>Drag and drop a mammogram here</div>
        <div style='color:#8b97b0; font-size:13px; margin-top:2px;'>Limit 200MB per file • PNG, JPG, JPEG</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload mammogram", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)

if uploaded_file:
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; gap:10px;
                    background:#182238; border:1px solid #25304a;
                    padding:10px 16px; border-radius:10px;
                    color:#e6ebf5; font-size:13px; margin-bottom:14px;'>
          <span style='color:#22d3ee;'>📄</span> <strong>{uploaded_file.name}</strong>
          <span style='color:#8b97b0; margin-left:4px;'>{uploaded_file.size // 1024} KB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── 2. Clinical inputs ──────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    density = st.selectbox("Breast Density", list(DENSITY_MAP.keys()))
with col2:
    view = st.selectbox("View Position", list(VIEW_MAP.keys()))

run_btn = st.button(
    "🔍 Run Classification",
    type="primary",
    use_container_width=True,
    disabled=model is None,
)

# ── 3. Run inference + Grad-CAM ─────────────────────────
if run_btn:
    if not uploaded_file:
        st.warning("Please upload a mammogram image first.")
    else:
        with st.spinner("Analysing mammogram…"):
            import numpy as np
            from core.inference import predict_single
            from core.preprocessing import val_transform, build_clinical
            from core.gradcam import get_gradcam_outputs

            pil_img = Image.open(uploaded_file)
            result = predict_single(model, pil_img, density, view, threshold)

            img_np = np.array(pil_img.convert("RGB"))
            img_t = val_transform(image=img_np)["image"]
            clin_t = torch.from_numpy(build_clinical(density, view))
            processed_img, heatmap_img, overlay_img = get_gradcam_outputs(model, img_t, clin_t)

            st.session_state.result = result
            st.session_state.orig_img = pil_img
            st.session_state.processed_img = processed_img
            st.session_state.heatmap_img = heatmap_img
            st.session_state.overlay_img = overlay_img
            st.session_state.result_ready = True

# ── 4. Results ──────────────────────────────────────────
if st.session_state.result_ready:
    result = st.session_state.result

    st.markdown("## 📷 Image & Heatmap")
    c1, c2, c3, c4 = st.columns(4)
    c1.image(st.session_state.orig_img, caption="Original", use_container_width=True)
    c2.image(st.session_state.processed_img, caption="Processed", use_container_width=True)
    c3.image(st.session_state.overlay_img, caption="Overlay", use_container_width=True)
    c4.image(st.session_state.heatmap_img, caption="Heatmap", use_container_width=True)

    st.markdown("---")
    st.markdown("## 🔬 Analysis Results")

    # Show whichever class has the higher raw probability, paired with that
    # probability as the displayed confidence — so "Normal" is never shown
    # while "Abnormal" is actually the more likely read. This intentionally
    # leans toward flagging "Abnormal": in screening, a missed abnormal case
    # is far costlier than a normal case sent for extra review.
    predicted_label = 0 if result["prob_normal"] >= result["prob_abnormal"] else 1
    label_txt = "Normal" if predicted_label == 0 else "Abnormal"
    accent = "#34d399" if predicted_label == 0 else "#fb7185"
    glow = "rgba(52,211,153,.25)" if predicted_label == 0 else "rgba(251,113,133,.25)"
    arrow = "↑" if predicted_label == 0 else "↓"
    conf = max(result["prob_normal"], result["prob_abnormal"]) * 100

    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg, #131a2e 0%, #182238 100%);
                    border:1px solid #25304a; border-left:4px solid {accent};
                    border-radius:14px; padding:22px 26px; margin-top:6px;
                    box-shadow: 0 12px 40px {glow};'>
          <div style='color:#8b97b0; font-size:12px; letter-spacing:1.5px;
                      text-transform:uppercase; font-weight:600;'>Prediction</div>
          <h2 style='color:{accent}; font-size:2.4rem; margin:6px 0 2px;
                     font-weight:800; letter-spacing:.3px;'>{label_txt}</h2>
          <div style='color:{accent}; font-size:1.05rem; font-weight:600;'>
            {arrow} {conf:.1f}% confidence
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Details"):
        st.write(
            {
                "Probability — Abnormal": f"{result['prob_abnormal']*100:.1f}%",
                "Probability — Normal": f"{result['prob_normal']*100:.1f}%",
                "Decision threshold": f"{result['threshold']:.3f}",
                "Breast density": density,
                "View position": view,
            }
        )

    st.info(
        "🩺 This is an AI research tool for thesis demonstration purposes only — "
        "it is **not** a medical diagnosis. Always consult a qualified radiologist."
    )
else:
    st.markdown(
        """
        <div style='border:1px dashed #25304a; border-radius:14px;
                    padding:28px; text-align:center; margin-top:10px;
                    color:#8b97b0; background:rgba(19,26,46,.4);'>
          <div style='font-size:28px; margin-bottom:8px;'>🔬</div>
          Upload a mammogram and click <strong style='color:#e6ebf5;'>Run Classification</strong>
          to see the Grad-CAM visualisation and prediction results.
        </div>
        """,
        unsafe_allow_html=True,
    )
