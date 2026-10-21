# TruthLens – Fake News Detector 🔍📰

TruthLens is a simple, modern Python NLP web application designed to detect fake news articles using Machine Learning. It combines **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization** and **Logistic Regression** within a responsive, blue-and-white web interface built with **Flask**, **Bootstrap 5**, and clean vanilla JavaScript.

---

## 🚀 Key Features

- **Modern Blue-and-White UI**: Sleek, clean responsive design with custom SVG illustrations and glassmorphism elements.
- **Natural Language Processing (NLP)**: Converts raw journalistic text into numerical features using unigram and bigram TF-IDF statistical scoring.
- **Logistic Regression Classifier**: Fast, interpretable linear classification providing probability estimates.
- **Real-Time Interactive Analysis**: Instant predictions with loading spinner, dual probability meters, and responsive feedback without full page reloads.
- **Explainable Verdicts**: Contextual AI explanations highlighting detected language style, rhetorical markers, and confidence percentages.
- **Responsible AI Disclaimer**: Prominently informs users that machine learning predictions are probabilistic indicators, not definitive proof of factual truth.
- **Quick Sample Testing**: One-click sample loader for testing authentic and fake news articles instantly.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python 3.10+, Flask |
| **Machine Learning** | Scikit-learn, Pandas, NumPy, Joblib |
| **NLP Pipeline** | TF-IDF Vectorizer (`ngram_range=(1,2)`, `sublinear_tf=True`) |
| **Classifier** | Logistic Regression (`solver='lbfgs'`) |
| **Frontend** | HTML5, CSS3, JavaScript (Fetch API), Bootstrap 5, Bootstrap Icons |

---

## 📁 Project Structure

```
fake-news-detector/
│
├── app.py                  # Flask web application & prediction API routes
├── train_model.py          # Script to preprocess dataset, train TF-IDF & Logistic Regression
├── requirements.txt        # Python package dependencies
├── README.md               # Documentation and execution guide
│
├── data/
│   └── news_dataset.csv    # Labeled news dataset (contains balanced sample set)
│
├── models/                 # Saved artifacts directory (generated after training)
│   ├── model.joblib        # Serialized Logistic Regression model
│   └── vectorizer.joblib   # Serialized TF-IDF vectorizer
│
├── templates/
│   ├── index.html          # Clean, modern landing page with AI illustration
│   └── home.html           # Interactive news detection workbench
│
└── static/
    ├── style.css           # Custom styling (blue-and-white theme, cards, animations)
    └── script.js           # Frontend interactions, AJAX requests, sample loaders
```

---

## ⚙️ Step-by-Step Setup & Execution

### 1. Open Terminal and Navigate to Project Directory

```bash
cd C:\Users\Student\.gemini\antigravity\scratch\fake-news-detector
```

### 2. Install Required Dependencies

Install the necessary libraries using `pip`:

```bash
pip install -r requirements.txt
```

*(If multiple Python versions exist on Windows, you can run `py -3.12 -m pip install -r requirements.txt`)*

---

### 3. Train the Machine Learning Model

Run the training script to process the labeled dataset, evaluate classification accuracy, and save the model artifacts:

```bash
python train_model.py
```

#### What happens during training:
1. Loads the labeled dataset from `data/news_dataset.csv`.
2. Splits data into **80% training** and **20% testing** sets with stratified class balance.
3. Fits the **TF-IDF Vectorizer** (unigrams and bigrams).
4. Trains the **Logistic Regression** classifier.
5. Evaluates accuracy, precision, recall, and confusion matrix on test data.
6. Serializes `model.joblib` and `vectorizer.joblib` into the `models/` directory.

---

### 4. Run the Flask Web Application

Start the local Flask development server:

```bash
python app.py
```

You will see:
```text
==================================================
 TruthLens – Fake News Detector Server Starting 
 URL: http://127.0.0.1:5000
==================================================
 * Running on http://127.0.0.1:5000
```

---

### 5. Open the Website in Your Browser

Open any web browser (Chrome, Edge, Firefox) and navigate to:

👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

- Click **Get Started** to open the **Check Your News** workbench (`/home`).
- Paste any article or headline, or click **Load Sample** to instantly test sample texts.
- Click **Analyze News** to see the **REAL NEWS** or **FAKE NEWS** verdict, confidence score, and explanation.

---

## 📊 Using Public Labeled Datasets

TruthLens includes a starter dataset in `data/news_dataset.csv`. You can easily scale up model accuracy by downloading larger public datasets:

1. **Kaggle Fake News Dataset** (e.g., [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)):
   - Download `True.csv` and `Fake.csv`.
   - Combine them or place your CSV file at `data/news_dataset.csv`.
2. **WELFake Dataset** or **ISOT Dataset**:
   - Ensure the CSV contains a text column (`text` or `title` + `text`) and a label column (`label` or `target` with values `REAL`/`FAKE` or `1`/`0`).
3. Re-run `python train_model.py` to retrain the model on the expanded dataset.

---

## ⚠️ Important Responsible AI Disclaimer

> **Notice**: TruthLens uses statistical Natural Language Processing to detect patterns and vocabulary typical of authentic reporting versus sensational or misleading text. Machine learning models are probabilistic and can produce false positives or false negatives. A "REAL NEWS" or "FAKE NEWS" prediction is **not conclusive proof** of factual accuracy. Always verify critical news through primary documentation and independent fact-checking organizations.
