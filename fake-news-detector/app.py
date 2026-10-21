"""
TruthLens – Fake News Detection System
Flask Web Application
"""

import os
import re
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")

# Global model and vectorizer cache
model = None
vectorizer = None


def load_artifacts():
    """Attempts to load the trained model and TF-IDF vectorizer."""
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return True
        except Exception as e:
            print(f"[ERROR] Error loading model artifacts: {e}")
            model = None
            vectorizer = None
            return False
    return False


# Initial artifact load attempt at startup
load_artifacts()


def clean_text(text: str) -> str:
    """Preprocesses input text by normalizing whitespace and casing."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def generate_explanation(is_real: bool, confidence: float, text: str) -> str:
    """Generates a contextual, user-friendly explanation of the prediction."""
    word_count = len(text.split())
    sample_lower = text.lower()

    # Characteristic keyword checks for educational context
    sensational_words = ["shocking", "miracle", "secret", "exposed", "cure", "banned", "alien", "conspiracy", "covert"]
    found_sensational = [w for w in sensational_words if w in sample_lower]

    if is_real:
        base = (
            f"TruthLens classified this text as REAL NEWS with {confidence:.1f}% confidence. "
            "The linguistic structure displays formal reporting tone, standard journalistic syntax, "
            "and objective terminology consistent with verified press sources."
        )
    else:
        indicators = f" (detected signals like: '{', '.join(found_sensational[:3])}')" if found_sensational else ""
        base = (
            f"TruthLens classified this text as FAKE NEWS with {confidence:.1f}% confidence. "
            f"The article exhibits sensationalized wording{indicators}, unverified claims, "
            "or rhetorical patterns commonly observed in disinformation."
        )

    if word_count < 15:
        base += " Note: Analyzing a short snippet or single headline may yield lower certainty than full article bodies."

    return base


@app.route("/")
def index():
    """Renders the modern TruthLens landing page."""
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    return render_template("index.html", model_ready=model_ready)


@app.route("/home")
def home():
    """Renders the interactive news detection workbench."""
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    return render_template("home.html", model_ready=model_ready)


@app.route("/api/status", methods=["GET"])
def model_status():
    """Returns the operational status of the ML model."""
    global model, vectorizer
    if model is None or vectorizer is None:
        load_artifacts()
    ready = model is not None and vectorizer is not None
    return jsonify({
        "model_loaded": ready,
        "model_path": MODEL_PATH if ready else None,
        "message": "Model is active and ready for inference." if ready else "Model is not trained yet. Run 'python train_model.py'."
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Analyzes input news text using the TF-IDF Vectorizer and Logistic Regression classifier.
    Supports both JSON AJAX requests and standard HTML form POSTs.
    """
    global model, vectorizer
    if model is None or vectorizer is None:
        if not load_artifacts():
            msg = "Model files not found. Please train the model first by running 'python train_model.py'."
            if request.is_json:
                return jsonify({"status": "error", "message": msg}), 503
            return render_template("home.html", model_ready=False, error_message=msg)

    # Extract text from JSON or Form body
    if request.is_json:
        data = request.get_json(silent=True) or {}
        news_text = data.get("text", "")
    else:
        news_text = request.form.get("news_text", "")

    cleaned = clean_text(news_text)

    # Empty validation
    if not cleaned:
        msg = "Please enter or paste news article text to analyze."
        if request.is_json:
            return jsonify({"status": "error", "message": msg}), 400
        return render_template("home.html", model_ready=True, error_message=msg, submitted_text="")

    try:
        # Transform input using the loaded vectorizer
        features = vectorizer.transform([cleaned])

        # Predict class and probabilities
        pred_class = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        # In our mapping: 0 = FAKE, 1 = REAL
        is_real = (pred_class == 1)
        label = "REAL NEWS" if is_real else "FAKE NEWS"
        confidence = float(probabilities[1] * 100 if is_real else probabilities[0] * 100)

        explanation = generate_explanation(is_real, confidence, cleaned)
        disclaimer = (
            "Disclaimer: Predictions are generated by an NLP machine learning model for informational purposes. "
            "They represent statistical likelihood based on training data patterns and do not prove whether an article is true or false."
        )

        response_data = {
            "status": "success",
            "prediction": label,
            "is_real": is_real,
            "confidence": round(confidence, 1),
            "real_probability": round(float(probabilities[1] * 100), 1),
            "fake_probability": round(float(probabilities[0] * 100), 1),
            "explanation": explanation,
            "disclaimer": disclaimer,
            "word_count": len(cleaned.split()),
            "character_count": len(cleaned)
        }

        if request.is_json:
            return jsonify(response_data)
        else:
            return render_template("home.html", model_ready=True, result=response_data, submitted_text=cleaned)

    except Exception as e:
        msg = f"An error occurred during prediction: {str(e)}"
        if request.is_json:
            return jsonify({"status": "error", "message": msg}), 500
        return render_template("home.html", model_ready=True, error_message=msg, submitted_text=cleaned)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    debug = False if os.environ.get("PORT") else True
    print(f"\n==================================================")
    print(f" TruthLens – Fake News Detector Server Starting ")
    print(f" URL: http://{host}:{port}")
    print(f"==================================================\n")
    app.run(host=host, port=port, debug=debug)
