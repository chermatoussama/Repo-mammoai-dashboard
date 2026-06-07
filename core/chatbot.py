# ─────────────────────────────────────────────────────────
# core/chatbot.py  — Claude-powered Q&A about the current prediction
# ─────────────────────────────────────────────────────────
import os
import streamlit as st

_CLIENT = None
_CLIENT_ERROR = None


def get_client():
    """Lazily build the Anthropic client. Returns None if no API key is set."""
    global _CLIENT, _CLIENT_ERROR
    if _CLIENT is not None or _CLIENT_ERROR is not None:
        return _CLIENT

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _CLIENT_ERROR = "missing_key"
        return None

    try:
        from anthropic import Anthropic
        _CLIENT = Anthropic(api_key=api_key)
    except Exception as exc:
        _CLIENT_ERROR = str(exc)
        return None

    return _CLIENT


def build_system_prompt(result: dict | None) -> str:
    """Build a system prompt grounded in the current prediction (if any)."""
    base = (
        "You are a medical AI assistant helping a thesis student explain "
        "mammography classification results from a research dashboard "
        "(EfficientNet-B3 NoisyStudent Dual-Stream model trained on the "
        "VinDr-Mammo dataset).\n\n"
    )

    if result:
        # Mirror the dashboard's displayed label: whichever class has the
        # higher raw probability (so the chatbot never contradicts the screen).
        label = "Normal" if result["prob_normal"] >= result["prob_abnormal"] else "Abnormal"
        base += (
            "Current prediction context:\n"
            f"- Prediction: {label}\n"
            f"- Probability of Abnormal: {result['prob_abnormal']*100:.1f}%\n"
            f"- Probability of Normal: {result['prob_normal']*100:.1f}%\n"
            f"- Decision threshold: {result['threshold']:.3f}\n"
            f"- Breast density: {result.get('density', 'N/A')}\n"
            f"- View position: {result.get('view', 'N/A')}\n\n"
        )
    else:
        base += (
            "No image has been analysed yet, so there is no prediction to "
            "discuss. Encourage the user to upload a mammogram and run "
            "classification first, or answer general questions about the "
            "model and dataset.\n\n"
        )

    base += (
        "Answer clearly and concisely (a few sentences, plain language). "
        "Always remind the user this is an AI research tool, not a medical "
        "diagnosis, and that any clinical decision must involve a qualified "
        "radiologist. Do NOT invent clinical details that are not present "
        "in the context above."
    )
    return base


_DISCLAIMER = (
    "\n\n🩺 *Reminder: this is an AI research tool for a thesis project, "
    "not a medical diagnosis — any clinical decision must involve a "
    "qualified radiologist.*"
)


def local_answer(prompt: str, result: dict | None) -> str:
    """
    Rule-based fallback Q&A — answers from the current prediction context
    without calling an external API. Used whenever no Anthropic API key is
    configured, so the chatbot stays useful out of the box.
    """
    q = prompt.lower()

    if result is None:
        if any(w in q for w in ("cancer", "breast", "predict", "result", "abnormal", "normal")):
            return (
                "I don't have a prediction to discuss yet — please upload a "
                "mammogram and click **Run Classification** first, then ask "
                "me again about the result." + _DISCLAIMER
            )
        return (
            "I can answer questions about the model and dataset:\n\n"
            "**Model**: EfficientNet-B3 NoisyStudent, dual-stream architecture "
            "fusing the mammogram image with clinical metadata (breast density, "
            "view position).\n\n"
            "**Dataset**: VinDr-Mammo — 20,137 mammograms, where BI-RADS 1–2 "
            "is labelled Normal and BI-RADS 3+ is labelled Abnormal.\n\n"
            "**Explainability**: Grad-CAM highlights the image regions that "
            "most influenced the model's decision.\n\n"
            "Upload an image and run a classification, and I can discuss the "
            "specific result with you." + _DISCLAIMER
        )

    # Mirror the dashboard's displayed label (highest raw probability).
    label = "Normal" if result["prob_normal"] >= result["prob_abnormal"] else "Abnormal"
    prob_abn = result["prob_abnormal"] * 100
    prob_norm = result["prob_normal"] * 100
    winning_prob = max(prob_abn, prob_norm)
    density = result.get("density", "N/A")
    view = result.get("view", "N/A")

    if any(w in q for w in ("grad-cam", "gradcam", "heatmap", "highlight", "overlay")):
        return (
            "Grad-CAM (Gradient-weighted Class Activation Mapping) shows "
            "which regions of the mammogram the model focused on most when "
            "making its prediction — warmer colours (red/yellow) mean higher "
            "influence, cooler colours (blue) mean lower influence. The "
            "**Overlay** panel blends this heatmap onto the original image so "
            "you can see exactly where the model 'looked'. It explains the "
            "model's reasoning — it is not itself a clinical finding." + _DISCLAIMER
        )

    if any(w in q for w in ("cancer", "have breast", "patient have", "tumour", "tumor", "malignan")):
        if label == "Abnormal":
            return (
                f"The model classified this mammogram as **Abnormal**, with "
                f"{prob_abn:.1f}% estimated probability of an abnormal finding "
                f"(BI-RADS ≥ 3 pattern). This does **not** mean a cancer "
                f"diagnosis — it flags the image for further radiological "
                f"review." + _DISCLAIMER
            )
        return (
            f"The model classified this mammogram as **Normal**, with "
            f"{prob_norm:.1f}% estimated probability of a normal finding "
            f"(BI-RADS 1–2 pattern). This is a screening-level estimate, not "
            f"a clinical clearance." + _DISCLAIMER
        )

    if any(w in q for w in ("why", "predict", "reason", "because")):
        return (
            f"The model leaned towards **{label}** because its visual stream "
            f"(EfficientNet-B3) and clinical stream (density: **{density}**, "
            f"view: **{view}**) jointly produced a probability of "
            f"{winning_prob:.1f}% for that class versus "
            f"{100 - winning_prob:.1f}% for the alternative. The Grad-CAM "
            f"overlay shows which regions of the image drove that decision "
            f"the most." + _DISCLAIMER
        )

    if any(w in q for w in ("screen", "doctor", "radiologist", "next step", "should i")):
        if label == "Abnormal":
            return (
                "Since the model flagged this image as **Abnormal**, the "
                "appropriate next step would be to have it reviewed by a "
                "qualified radiologist for a definitive assessment and "
                "possible follow-up imaging — this tool is a screening aid, "
                "not a diagnostic authority." + _DISCLAIMER
            )
        return (
            "The model leaned towards **Normal** for this image, but routine "
            "screening schedules and any clinical concerns should always be "
            "guided by a qualified radiologist or physician, regardless of "
            "this tool's output." + _DISCLAIMER
        )

    if any(w in q for w in ("confiden", "probability", "percent", "score", "sure")):
        return (
            f"The model is most confident in **{label}**, at "
            f"{winning_prob:.1f}% (Abnormal: {prob_abn:.1f}% · "
            f"Normal: {prob_norm:.1f}%). The decision threshold used was "
            f"{result['threshold']:.3f}." + _DISCLAIMER
        )

    return (
        f"For this mammogram (density **{density}**, view **{view}**), the "
        f"model's prediction is **{label}** with {winning_prob:.1f}% "
        f"confidence (Abnormal: {prob_abn:.1f}% · Normal: {prob_norm:.1f}%). "
        f"Feel free to ask me *why* it predicted this, what the *Grad-CAM* "
        f"heatmap shows, or whether *further screening* is advisable." + _DISCLAIMER
    )


def handle_chat(prompt: str, result: dict | None):
    """Append the user message, get a reply (Claude if configured, otherwise
    a local rule-based answer grounded in the current prediction), and rerun."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    client = get_client()
    if client is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": local_answer(prompt, result),
        })
        st.rerun()
        return

    system = build_system_prompt(result)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m["role"] in ("user", "assistant")
            ],
        )
        reply = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip() or "I wasn't able to generate a response — please try again."
    except Exception as exc:
        reply = f"⚠️ Couldn't reach the AI assistant right now ({exc})."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
