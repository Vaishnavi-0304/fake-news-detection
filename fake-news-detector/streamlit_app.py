"""
TruthLens – Fake News Detection System
Streamlit Web Application
"""

import os
import re
import joblib
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="TruthLens – Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Blue-and-White Theme Styling
st.markdown("""
<style>
    /* Global Styles */
    .main-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1E40AF;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #475569;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        background-color: #EEF2FF;
        color: #2563EB;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
        margin-bottom: 12px;
    }
    .verdict-box-real {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        padding: 24px;
        border-radius: 16px;
        color: #065F46;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .verdict-box-fake {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        padding: 24px;
        border-radius: 16px;
        color: #991B1B;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .verdict-header {
        font-size: 1.8rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .explanation-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 15px;
        color: #334155;
    }
    .disclaimer-card {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")


@st.cache_resource
def load_ml_artifacts():
    """Loads and caches the model and TF-IDF vectorizer."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            v = joblib.load(VECTORIZER_PATH)
            return m, v
        except Exception as e:
            st.error(f"Error loading model artifacts: {e}")
            return None, None
    return None, None


model, vectorizer = load_ml_artifacts()

# Sample articles for testing
SAMPLE_REAL = (
    "NASA announced that the James Webb Space Telescope detected atmospheric water vapor "
    "and methane signatures on a rocky exoplanet orbiting within the habitable zone of a red dwarf "
    "star 120 light years away. Peer-reviewed spectroscopic measurements verified the presence "
    "of chemical absorption lines corresponding to H2O molecules."
)

SAMPLE_FAKE = (
    "Secret leaked government files prove that world leaders have met inside a hollow moon base "
    "with extraterrestrial reptilian overlords. An anonymous whistleblower claims that 5G frequencies "
    "are secretly spraying synthetic pathogens into city drinking supplies to force everyone into "
    "instant mind-control compliance."
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=64)
    st.title("TruthLens AI")
    st.caption("Fake News Detection System")
    st.markdown("---")
    st.subheader("Technology Stack")
    st.markdown("- **Framework**: Streamlit / Flask")
    st.markdown("- **NLP**: TF-IDF (Unigrams & Bigrams)")
    st.markdown("- **Classifier**: Logistic Regression")
    st.markdown("- **Serialization**: Joblib")
    st.markdown("---")
    st.subheader("Model Status")
    if model is not None and vectorizer is not None:
        st.success("Model: Active & Loaded")
    else:
        st.warning("Model: Not Found (Run train_model.py)")
    st.markdown("---")
    st.caption("Responsible AI: Probabilistic NLP estimate for informational research.")

# Main Header
st.markdown('<div class="badge-pill">Natural Language Processing & Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">TruthLens – Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Evaluate the credibility of news articles and headlines using TF-IDF feature analysis and Logistic Regression.</p>', unsafe_allow_html=True)

# Sample Buttons
col_sample1, col_sample2, col_clear = st.columns([1.5, 1.5, 3])

if "article_input" not in st.session_state:
    st.session_state.article_input = ""

with col_sample1:
    if st.button("Load Real News Sample", use_container_width=True):
        st.session_state.article_input = SAMPLE_REAL

with col_sample2:
    if st.button("Load Fake News Sample", use_container_width=True):
        st.session_state.article_input = SAMPLE_FAKE

with col_clear:
    if st.button("Clear Input", use_container_width=False):
        st.session_state.article_input = ""

# Input Text Area
article_text = st.text_area(
    "News Article or Headline:",
    value=st.session_state.article_input,
    height=180,
    placeholder="Paste news headline or paragraph here..."
)

# Word and Character Counter
words = len(article_text.strip().split()) if article_text.strip() else 0
chars = len(article_text)
st.caption(f"{words} words | {chars} characters")

# Analyze Button
analyze_clicked = st.button("🔍 Analyze News", type="primary", use_container_width=False)

if analyze_clicked:
    cleaned = article_text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if not cleaned:
        st.error("Please enter or paste news article text before analyzing.")
    elif model is None or vectorizer is None:
        st.error("Model artifacts not found. Please ensure 'models/model.joblib' and 'models/vectorizer.joblib' are present.")
    else:
        with st.spinner("Analyzing text patterns with TF-IDF Vectorizer and Logistic Regression..."):
            features = vectorizer.transform([cleaned])
            pred_class = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]

            is_real = (pred_class == 1)
            label = "REAL NEWS" if is_real else "FAKE NEWS"
            confidence = float(probabilities[1] * 100 if is_real else probabilities[0] * 100)
            real_prob = float(probabilities[1] * 100)
            fake_prob = float(probabilities[0] * 100)

            # Contextual Explanation
            sensational_words = ["shocking", "miracle", "secret", "exposed", "cure", "banned", "alien", "conspiracy"]
            found = [w for w in sensational_words if w in cleaned.lower()]

            if is_real:
                explanation = (
                    f"TruthLens classified this article as **REAL NEWS** with **{confidence:.1f}% confidence**. "
                    "The linguistic structure displays formal reporting syntax, institutional terminology, "
                    "and objective phrasing typical of verified journalism."
                )
                box_class = "verdict-box-real"
                icon = "✅"
            else:
                indicators = f" (detected sensational terms: *{', '.join(found[:3])}*)" if found else ""
                explanation = (
                    f"TruthLens classified this article as **FAKE NEWS** with **{confidence:.1f}% confidence**. "
                    f"The text exhibits high emotional polarity{indicators}, unverified claims, or sensationalized rhetoric."
                )
                box_class = "verdict-box-fake"
                icon = "⚠️"

            if words < 15:
                explanation += " *(Note: Short snippets or headlines may yield lower certainty than full articles).* "

            # Render Verdict Card
            st.markdown(f"""
            <div class="{box_class}">
                <div class="verdict-header">
                    <span>{icon}</span>
                    <span>{label}</span>
                </div>
                <p style="margin-top: 8px; font-size: 1.1rem; margin-bottom: 0;">
                    Model Confidence: <strong>{confidence:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability Metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(label="Real News Probability", value=f"{real_prob:.1f}%")
                st.progress(real_prob / 100.0)
            with col_m2:
                st.metric(label="Fake News Probability", value=f"{fake_prob:.1f}%")
                st.progress(fake_prob / 100.0)

            # Explanation
            st.markdown(f"""
            <div class="explanation-card">
                <strong>💡 Prediction Explanation:</strong><br>
                {explanation}
            </div>
            """, unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
            <div class="disclaimer-card">
                <strong>⚠️ Responsible AI Notice:</strong> Predictions are generated by an NLP machine learning model for informational purposes. They represent statistical likelihood based on training data patterns and do not prove whether an article is true or false. Always verify critical news with primary sources.
            </div>
            """, unsafe_allow_html=True)
