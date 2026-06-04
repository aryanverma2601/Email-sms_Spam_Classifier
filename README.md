# Email/SMS Spam Classifier

A machine learning model that classifies emails and SMS messages as **spam** or **ham (not spam)** using Natural Language Processing.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-NaiveBayes-orange?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red?style=flat-square) ![Accuracy](https://img.shields.io/badge/Accuracy-97.10%25-brightgreen?style=flat-square) ![Precision](https://img.shields.io/badge/Precision-100%25-brightgreen?style=flat-square)

---

## Overview

This project builds a text classification pipeline using **Multinomial Naive Bayes** and **TF-IDF Vectorization** to detect spam in real-world email and SMS datasets. The model is trained and evaluated in a Jupyter Notebook, with an interactive **Streamlit** app for real-time classification.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 97.10% |
| Precision | 100% |

> High precision means zero false positives — no legitimate messages are incorrectly flagged as spam.

---

## Features

- Binary classification: **Spam** vs **Ham**
- Full NLP preprocessing pipeline — tokenization, stemming, stopword removal
- TF-IDF feature extraction
- Multinomial Naive Bayes classifier
- Interactive Streamlit web app for real-time message classification

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| NLP | NLTK |
| ML | scikit-learn (TF-IDF, Multinomial Naive Bayes) |
| Notebook | Jupyter |
| Deployment | Streamlit |

---

## Project Structure

```
Email-sms_Spam_Classifier/
│
├── spam-classifier/
│   ├── app.py                            # Streamlit app entry point
│   ├── model.pkl                         # Trained Naive Bayes model
│   ├── vectorizer.pkl                    # Fitted TF-IDF vectorizer
├── Email_sms_spam_classifier.ipynb       # Full EDA, training & evaluation
│
└── README.md
```

---

## How It Works

### 1. Preprocessing
Raw messages go through an NLP cleaning pipeline built with **NLTK**:
- Lowercasing and punctuation removal
- **Tokenization** — splitting text into individual tokens
- **Stopword removal** — filtering out common words like "the", "is", "and"
- **Stemming** — reducing words to their root form (e.g. "winning" → "win") using Porter Stemmer

### 2. Feature Extraction
Cleaned tokens are converted to numerical features using **TF-IDF Vectorization**, which weighs terms by their frequency and relative importance across the corpus.

### 3. Classification
A **Multinomial Naive Bayes** classifier is trained on the TF-IDF matrix. Naive Bayes is well-suited for text classification — fast, interpretable, and strong on sparse, high-dimensional data.

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/aryanverma2601/Email-sms_Spam_Classifier.git
cd Email-sms_Spam_Classifier

# Install dependencies
pip install nltk scikit-learn streamlit pandas numpy

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Run the Streamlit App

```bash
cd spam-classifier
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Paste any email or SMS message and get an instant spam/ham prediction.

### Explore the Notebook

Open `Email_sms_spam_classifier.ipynb` in Jupyter to walk through the full pipeline — data cleaning, EDA, model training, and evaluation.

```bash
jupyter notebook spam-classifier/Email_sms_spam_classifier.ipynb
```

---

## Author

**Aryan Verma**  
[GitHub](https://github.com/aryanverma2601) · [LinkedIn](https://linkedin.com/in/aryan-verma) · aryanv380@gmail.com
