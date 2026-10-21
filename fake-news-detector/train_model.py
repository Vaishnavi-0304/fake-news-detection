"""
TruthLens – Fake News Detection System
Model Training Script
Trains a TF-IDF Vectorizer and Logistic Regression classifier on labeled news data.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "news_dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")


def load_dataset(file_path):
    """
    Loads and validates the labeled fake-news dataset.
    Expects a CSV with text and label columns.
    """
    if not os.path.exists(file_path):
        print(f"\n[ERROR] Dataset not found at: {file_path}")
        print("Please ensure your dataset is placed at 'data/news_dataset.csv'.")
        print("Required CSV columns: 'text' (or 'title'+'text') and 'label' (REAL/FAKE or 1/0).")
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}")
        return None

    # Handle various column name conventions
    text_col = None
    for candidate in ["text", "article", "content", "news", "body"]:
        if candidate in df.columns:
            text_col = candidate
            break

    # If title exists and text exists, combine them for richer context
    if "title" in df.columns and text_col:
        df["full_text"] = df["title"].fillna("") + " " + df[text_col].fillna("")
        text_col = "full_text"
    elif not text_col and "title" in df.columns:
        text_col = "title"

    if not text_col:
        print("[ERROR] Could not find a text column in dataset. Expected 'text', 'title', or 'content'.")
        return None

    label_col = None
    for candidate in ["label", "target", "class", "is_fake"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if not label_col:
        print("[ERROR] Could not find a label column in dataset. Expected 'label' or 'target'.")
        return None

    # Clean missing values
    df = df.dropna(subset=[text_col, label_col]).copy()

    # Normalize labels:
    # We standardize to: 1 = REAL, 0 = FAKE
    def normalize_label(val):
        val_str = str(val).strip().upper()
        if val_str in ["1", "REAL", "TRUE", "GENUINE", "RELIABLE"]:
            return 1
        elif val_str in ["0", "FAKE", "FALSE", "UNRELIABLE"]:
            return 0
        else:
            return None

    df["standard_label"] = df[label_col].apply(normalize_label)
    df = df.dropna(subset=["standard_label"]).copy()
    df["standard_label"] = df["standard_label"].astype(int)

    if len(df) < 10:
        print(f"[ERROR] Dataset contains too few rows ({len(df)}). At least 10 labeled samples are required.")
        return None

    return df[[text_col, "standard_label"]].rename(columns={text_col: "text", "standard_label": "label"})


def train():
    """Trains the TF-IDF Vectorizer + Logistic Regression pipeline and saves artifacts."""
    print("=" * 60)
    print(" TruthLens – Fake News Detector: Model Training ")
    print("=" * 60)

    print(f"[*] Loading dataset from: {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    if df is None:
        sys.exit(1)

    total_samples = len(df)
    real_count = (df["label"] == 1).sum()
    fake_count = (df["label"] == 0).sum()
    print(f"[+] Loaded {total_samples} samples: {real_count} REAL, {fake_count} FAKE.")

    # Split dataset: 80% Train, 20% Test
    test_size = 0.2 if total_samples >= 20 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

    print(f"[*] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # TF-IDF Feature Extraction
    print("[*] Fitting TF-IDF Vectorizer (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression
    print("[*] Training Logistic Regression classifier...")
    model = LogisticRegression(
        C=5.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "-" * 40)
    print(f" Model Accuracy: {accuracy * 100:.2f}%")
    print("-" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"], digits=4, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted FAKE   Predicted REAL")
    print(f"  Actual FAKE       {cm[0][0]:<14}  {cm[0][1]:<14}")
    print(f"  Actual REAL       {cm[1][0]:<14}  {cm[1][1]:<14}")

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\n[*] Saving model and vectorizer...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[+] Model saved to: {MODEL_PATH}")
    print(f"[+] Vectorizer saved to: {VECTORIZER_PATH}")
    print("\n Training complete! You can now run the Flask web app with: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
