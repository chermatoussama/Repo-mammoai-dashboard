# ─────────────────────────────────────────────────────────
# core/chatbot.py  — Claude-powered Q&A about the current prediction
# Falls back to an enhanced rule-based system when no API key is set.
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
        "You have deep knowledge about:\n"
        "- Breast cancer detection and mammography\n"
        "- Deep learning in medical imaging (CNNs, EfficientNet, transfer learning)\n"
        "- The VinDr-Mammo dataset (20,137 mammograms, BI-RADS classification)\n"
        "- Grad-CAM explainability technique\n"
        "- BI-RADS scoring system (1-5)\n"
        "- Breast density categories (A, B, C, D)\n"
        "- Mammogram views (CC = Craniocaudal, MLO = Mediolateral Oblique)\n"
        "- Model evaluation metrics (AUC, sensitivity, specificity, F1)\n"
        "- Clinical workflow for breast cancer screening\n\n"
    )

    if result:
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
            "No image has been analysed yet. You can still answer general questions "
            "about breast cancer, mammography, the model architecture, the dataset, "
            "Grad-CAM, BI-RADS, or any other related topic.\n\n"
        )

    base += (
        "Answer clearly and concisely in the same language the user uses "
        "(Arabic or English). "
        "Always remind the user this is an AI research tool, not a medical "
        "diagnosis, and that any clinical decision must involve a qualified "
        "radiologist. Do NOT invent clinical details not present in the context."
    )
    return base


_DISCLAIMER = (
    "\n\n🩺 *تذكير: هذا أداة بحثية للأطروحة، وليس تشخيصاً طبياً — "
    "يجب أن يتضمن أي قرار سريري طبيباً متخصصاً.*"
)

_DISCLAIMER_EN = (
    "\n\n🩺 *Reminder: this is an AI research tool for a thesis project, "
    "not a medical diagnosis — any clinical decision must involve a "
    "qualified radiologist.*"
)

# ─────────────────────────────────────────────────────────
# Enhanced local fallback — handles a wide range of questions
# ─────────────────────────────────────────────────────────

_KNOWLEDGE_BASE = {
    # ── Dataset ──────────────────────────────────────────
    "vindr": (
        "**VinDr-Mammo** هو مجموعة بيانات ماموجرام فيتنامية تحتوي على **20,137 صورة** "
        "من 4,000 مريضة، مع تصنيفات BI-RADS من 5 أطباء أشعة خبراء.\n\n"
        "- **BI-RADS 1-2** → Normal (طبيعي)\n"
        "- **BI-RADS 3-5** → Abnormal (غير طبيعي)\n\n"
        "تم نشر المجموعة عام 2022 وهي من أكبر مجموعات بيانات الماموجرام المفتوحة المصدر."
    ),
    "dataset": (
        "المجموعة المستخدمة هي **VinDr-Mammo** — 20,137 ماموجرام مقسّمة إلى:\n"
        "- 70% تدريب · 15% تحقق · 15% اختبار\n"
        "- ثنائي التصنيف: Normal (BI-RADS 1-2) vs Abnormal (BI-RADS 3+)"
    ),
    # ── Model Architecture ────────────────────────────────
    "efficientnet": (
        "**EfficientNet-B3 NoisyStudent** هو نموذج CNN متقدم من Google:\n\n"
        "- **B3**: الإصدار الثالث في عائلة EfficientNet (أكبر من B0، أصغر من B7)\n"
        "- **NoisyStudent**: تقنية تدريب شبه-إشرافي تضيف ضوضاء أثناء التدريب لتحسين التعميم\n"
        "- **1536 ميزة** تُستخرج من آخر طبقة conv_head\n"
        "- مُدرَّب مسبقاً على ImageNet ثم fine-tuned على VinDr-Mammo"
    ),
    "dual stream": (
        "**المعمارية ثنائية التدفق (Dual-Stream Architecture)**:\n\n"
        "```\n"
        "صورة الماموجرام → EfficientNet-B3 → 1536 ميزة بصرية\n"
        "                                              ↓\n"
        "بيانات سريرية  → MLP (6→64→128)  → 128 ميزة سريرية\n"
        "                                              ↓\n"
        "               Fusion: Linear(1664→512→128→2)\n"
        "                                              ↓\n"
        "                     Normal / Abnormal\n"
        "```\n\n"
        "**البيانات السريرية** = كثافة الثدي (A/B/C/D) + زاوية التصوير (CC/MLO) → "
        "متجه one-hot بـ 6 أبعاد"
    ),
    "architecture": (
        "المعمارية تتكون من تدفقين:\n\n"
        "1. **Visual Stream**: EfficientNet-B3 NS يعالج صورة الماموجرام → 1536 ميزة\n"
        "2. **Clinical Stream**: MLP يعالج كثافة الثدي وزاوية التصوير → 128 ميزة\n"
        "3. **Fusion**: دمج 1664 ميزة → تصنيف نهائي\n\n"
        "الفكرة: الموديل لا يعتمد على الصورة وحدها، بل يدمج السياق السريري تماماً "
        "كما يفعل الطبيب الأشعة."
    ),
    # ── Grad-CAM ──────────────────────────────────────────
    "grad": (
        "**Grad-CAM (Gradient-weighted Class Activation Mapping)**:\n\n"
        "تقنية XAI (Explainable AI) تُظهر *أين* نظر الموديل في الصورة:\n\n"
        "1. تُحسب تدرجات الـ gradient للـ logit المستهدف (Abnormal)\n"
        "2. تُوزَّن خرائط الفعالية (feature maps) بمتوسط هذه التدرجات\n"
        "3. تُحوَّل إلى خريطة حرارية (heatmap) بألوان COLORMAP_JET\n\n"
        "- **أحمر/أصفر** = المناطق الأكثر تأثيراً في القرار\n"
        "- **أزرق** = مناطق أقل أهمية\n\n"
        "الطبقة المستهدفة: `model.backbone.conv_head` (آخر طبقة في EfficientNet)"
    ),
    "heatmap": (
        "الخريطة الحرارية (Heatmap) تُنتج 4 صور في الداشبورد:\n\n"
        "| الصورة | الوصف |\n"
        "|---|---|\n"
        "| Original | الصورة الأصلية كما رُفعت |\n"
        "| Processed | بعد resize إلى 512×512 وتطبيع ImageNet |\n"
        "| Heatmap | خريطة Grad-CAM النقية بألوان JET |\n"
        "| Overlay | دمج الهيتماب مع الصورة (alpha=0.45) |"
    ),
    # ── BI-RADS ───────────────────────────────────────────
    "birads": (
        "**نظام BI-RADS** (Breast Imaging-Reporting and Data System):\n\n"
        "| الدرجة | المعنى | في موديلنا |\n"
        "|---|---|---|\n"
        "| BI-RADS 1 | طبيعي تماماً | Normal |\n"
        "| BI-RADS 2 | إيجابي حميد | Normal |\n"
        "| BI-RADS 3 | ربما حميد — متابعة | **Abnormal** |\n"
        "| BI-RADS 4 | مشتبه به — خزعة | **Abnormal** |\n"
        "| BI-RADS 5 | سرطان مرجّح جداً | **Abnormal** |"
    ),
    # ── Density ───────────────────────────────────────────
    "density": (
        "**كثافة الثدي** تؤثر على دقة الكشف:\n\n"
        "| الفئة | الوصف | التأثير |\n"
        "|---|---|---|\n"
        "| Density A | دهني بالكامل | أسهل كشفاً |\n"
        "| Density B | كثافة متناثرة | جيد |\n"
        "| Density C | كثافة غير متجانسة | أصعب |\n"
        "| Density D | كثيف جداً | الأصعب — قد يخفي الأورام |\n\n"
        "الموديل يأخذ الكثافة كمدخل سريري لتحسين الدقة في الحالات الصعبة."
    ),
    # ── Threshold ─────────────────────────────────────────
    "threshold": (
        "**عتبة القرار (Decision Threshold) = 0.60**\n\n"
        "لماذا 0.60 وليس 0.50؟\n\n"
        "في الفحص الطبي (Screening)، **تفويت حالة سرطانية أخطر بكثير** "
        "من إرسال حالة طبيعية للمراجعة.\n\n"
        "- العتبة الأمثل لـ F1 كانت ~0.68\n"
        "- خفّضناها إلى 0.60 لزيادة الـ **Sensitivity** (حساسية الكشف)\n"
        "- هذا يعني: إذا كان احتمال Abnormal ≥ 60% → نصنّفه Abnormal\n\n"
        "المعادلة: **Sensitivity > Specificity** في تطبيقات الفحص الوقائي"
    ),
    # ── Training ──────────────────────────────────────────
    "train": (
        "**تدريب الموديل:**\n\n"
        "- **Pre-training**: ImageNet (transfer learning)\n"
        "- **Fine-tuning**: VinDr-Mammo\n"
        "- **تحسين**: Adam optimizer\n"
        "- **Regularization**: Dropout (0.3) + LayerNorm + DropPath\n"
        "- **Data Augmentation**: Albumentations (flip, rotate, brightness)\n"
        "- **Loss**: CrossEntropy مع class weights لمعالجة عدم التوازن\n"
        "- **الجهاز**: GPU (CUDA) أثناء التدريب، CPU عند النشر"
    ),
    # ── View positions ────────────────────────────────────
    "cc": (
        "**CC (Craniocaudal)** — التصوير من الأعلى إلى الأسفل:\n\n"
        "- يُظهر الثدي من المنظر الأمامي\n"
        "- مناسب لرؤية الكتلة الداخلية والخارجية\n\n"
        "**MLO (Mediolateral Oblique)** — زاوية مائلة:\n\n"
        "- الزاوية الأكثر استخداماً في الفحص الروتيني\n"
        "- تُظهر أكبر قدر من أنسجة الثدي والإبط"
    ),
    "mlo": (
        "**MLO (Mediolateral Oblique)** هي زاوية التصوير المائلة:\n\n"
        "- تُصوَّر الصورة بزاوية ~45 درجة\n"
        "- تُظهر أكبر جزء من أنسجة الثدي\n"
        "- تشمل العضلة الصدرية والإبط\n"
        "- الأكثر استخداماً في برامج الفحص الروتيني (Screening)"
    ),
    # ── Technology stack ──────────────────────────────────
    "streamlit": (
        "**Streamlit** هو إطار عمل Python لبناء تطبيقات الويب للبيانات:\n\n"
        "- كود Python بحت — لا HTML أو JavaScript\n"
        "- مناسب لعروض Data Science و ML\n"
        "- يدعم الرفع، المخططات التفاعلية، الشات\n"
        "- النشر على Streamlit Community Cloud مجاناً"
    ),
    "pytorch": (
        "**PyTorch** هو الإطار المستخدم لبناء وتشغيل الموديل:\n\n"
        "- `torch.no_grad()`: تعطيل حساب التدرجات أثناء الاستدلال لتسريع الأداء\n"
        "- `F.softmax()`: تحويل الـ logits إلى احتمالات (مجموعها 1)\n"
        "- `@st.cache_resource`: تحميل الموديل مرة واحدة فقط في الذاكرة"
    ),
    "timm": (
        "**timm (PyTorch Image Models)** هي مكتبة تحتوي على مئات النماذج:\n\n"
        "- استُخدمت لتحميل `tf_efficientnet_b3_ns` (NoisyStudent)\n"
        "- `num_classes=0`: نزع طبقة التصنيف الأصلية للاستخدام كـ Feature Extractor\n"
        "- `pretrained=False`: لأن الأوزان محمّلة من ملف `.pth` المخصص"
    ),
}


def _detect_language(text: str) -> str:
    """Simple Arabic/English detector."""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "ar" if arabic_chars > len(text) * 0.2 else "en"


def local_answer(prompt: str, result: dict | None) -> str:
    """
    Enhanced rule-based fallback Q&A — covers a wide range of questions
    about the model, dataset, technology, and current prediction.
    Used whenever no Anthropic API key is configured.
    """
    q = prompt.lower()
    lang = _detect_language(prompt)
    disc = _DISCLAIMER if lang == "ar" else _DISCLAIMER_EN

    # ── Search knowledge base first ─────────────────────
    for keyword, answer in _KNOWLEDGE_BASE.items():
        if keyword in q:
            return answer + disc

    # ── Prediction-specific answers ─────────────────────
    if result is None:
        if any(w in q for w in ("cancer", "سرطان", "predict", "تنبؤ", "result",
                                 "نتيجة", "abnormal", "normal", "طبيعي")):
            if lang == "ar":
                return (
                    "لا يوجد تحليل بعد — يرجى **رفع صورة ماموجرام** "
                    "والنقر على **Run Classification** أولاً، ثم اسألني عن النتيجة."
                    + disc
                )
            return (
                "No prediction yet — please **upload a mammogram** and click "
                "**Run Classification** first, then ask me about the result." + disc
            )

        if lang == "ar":
            return (
                "يمكنني الإجابة على أسئلة حول:\n\n"
                "- 🧠 **الموديل**: EfficientNet-B3 NoisyStudent، المعمارية ثنائية التدفق\n"
                "- 📊 **المجموعة**: VinDr-Mammo، BI-RADS، كثافة الثدي\n"
                "- 🔍 **Grad-CAM**: كيف يشرح الموديل قراره\n"
                "- ⚙️ **التقنيات**: PyTorch، Streamlit، timm، Albumentations\n"
                "- 🏥 **الطب**: نظام BI-RADS، كثافة الثدي، أوضاع التصوير\n\n"
                "ارفع صورة وشغّل التحليل لمناقشة النتيجة." + disc
            )
        return (
            "I can answer questions about:\n\n"
            "- 🧠 **Model**: EfficientNet-B3 NoisyStudent, dual-stream architecture\n"
            "- 📊 **Dataset**: VinDr-Mammo, BI-RADS, breast density\n"
            "- 🔍 **Grad-CAM**: how the model explains its decision\n"
            "- ⚙️ **Tech**: PyTorch, Streamlit, timm, Albumentations\n"
            "- 🏥 **Clinical**: BI-RADS scoring, density categories, view positions\n\n"
            "Upload an image and run classification to discuss the prediction." + disc
        )

    # ── With a result ───────────────────────────────────
    label = "Normal" if result["prob_normal"] >= result["prob_abnormal"] else "Abnormal"
    label_ar = "طبيعي" if label == "Normal" else "غير طبيعي"
    prob_abn = result["prob_abnormal"] * 100
    prob_norm = result["prob_normal"] * 100
    winning_prob = max(prob_abn, prob_norm)
    density = result.get("density", "N/A")
    view = result.get("view", "N/A")

    if any(w in q for w in ("cancer", "سرطان", "have", "patient", "لديه", "مريض")):
        if lang == "ar":
            if label == "Abnormal":
                return (
                    f"صنّف الموديل هذا الماموجرام كـ **غير طبيعي (Abnormal)** "
                    f"باحتمال {prob_abn:.1f}%. هذا **لا يعني** تشخيص سرطان — "
                    f"بل يعني أن الصورة تستوجب مراجعة طبيب أشعة متخصص." + disc
                )
            return (
                f"صنّف الموديل هذا الماموجرام كـ **طبيعي (Normal)** "
                f"باحتمال {prob_norm:.1f}%. هذا تقدير للفحص الأولي، "
                f"وليس تصفية سريرية نهائية." + disc
            )
        if label == "Abnormal":
            return (
                f"The model classified this as **Abnormal** ({prob_abn:.1f}%). "
                f"This does **not** mean cancer — it flags the image for "
                f"radiological review." + disc
            )
        return (
            f"The model classified this as **Normal** ({prob_norm:.1f}%). "
            f"This is a screening-level estimate, not a clinical clearance." + disc
        )

    if any(w in q for w in ("why", "لماذا", "reason", "سبب", "because", "لأن")):
        if lang == "ar":
            return (
                f"اتجه الموديل نحو **{label_ar}** لأن:\n\n"
                f"- التدفق البصري (EfficientNet-B3) استخرج أنماطاً تُشير إلى {label_ar}\n"
                f"- التدفق السريري أخذ بعين الاعتبار: كثافة **{density}** وزاوية **{view}**\n"
                f"- الاحتمال النهائي: {winning_prob:.1f}% مقابل {100-winning_prob:.1f}%\n\n"
                f"انظر إلى صورة **Overlay** لترى المناطق التي ركّز عليها الموديل." + disc
            )
        return (
            f"The model leaned toward **{label}** because:\n\n"
            f"- Visual stream extracted patterns consistent with {label}\n"
            f"- Clinical stream used: density **{density}**, view **{view}**\n"
            f"- Final probability: {winning_prob:.1f}% vs {100-winning_prob:.1f}%\n\n"
            f"Check the **Overlay** image to see where the model focused." + disc
        )

    if any(w in q for w in ("confidence", "ثقة", "probability", "احتمال",
                             "percent", "نسبة", "score", "sure")):
        if lang == "ar":
            return (
                f"الموديل واثق من **{label_ar}** بنسبة **{winning_prob:.1f}%**\n\n"
                f"- احتمال Abnormal: {prob_abn:.1f}%\n"
                f"- احتمال Normal: {prob_norm:.1f}%\n"
                f"- العتبة المستخدمة: {result['threshold']:.3f}" + disc
            )
        return (
            f"The model is **{winning_prob:.1f}%** confident in **{label}**\n\n"
            f"- Abnormal: {prob_abn:.1f}% · Normal: {prob_norm:.1f}%\n"
            f"- Threshold used: {result['threshold']:.3f}" + disc
        )

    # ── Generic helpful fallback ─────────────────────────
    if lang == "ar":
        return (
            f"للماموجرام هذا (كثافة **{density}**، زاوية **{view}**)، "
            f"قرار الموديل: **{label_ar}** بنسبة ثقة {winning_prob:.1f}%.\n\n"
            f"يمكنك سؤالي عن:\n"
            f"- **لماذا** توصّل الموديل لهذه النتيجة\n"
            f"- ما تعني صورة **Grad-CAM**\n"
            f"- **الخطوة التالية** سريرياً\n"
            f"- أي سؤال تقني عن الموديل أو المجموعة" + disc
        )
    return (
        f"For this mammogram (density **{density}**, view **{view}**), "
        f"the model predicts **{label}** with {winning_prob:.1f}% confidence.\n\n"
        f"You can ask me:\n"
        f"- **Why** the model predicted this\n"
        f"- What the **Grad-CAM** heatmap shows\n"
        f"- What the **next clinical step** would be\n"
        f"- Any technical question about the model or dataset" + disc
    )


def handle_chat(prompt: str, result: dict | None):
    """Append the user message, get a reply (Claude if configured, otherwise
    an enhanced local rule-based answer), and rerun."""
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
            model="claude-sonnet-4-5",
            max_tokens=1024,
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
