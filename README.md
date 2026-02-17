# 📧 Spam Detection

The objective was to build a system capable of filtering unwanted messages by analyzing their textual content. Since machines cannot process raw text, this project focuses on the critical step of converting unstructured strings into numerical features using mathematical weighting.

## 🛠️ Key Features

* **Custom NLP Pipeline:** Includes text cleaning (lowercase conversion, punctuation removal, and tokenization) to handle over 5,500 real-world messages.
* **TF-IDF Vectorization:** Utilized **Term Frequency-Inverse Document Frequency** to calculate the importance of words relative to the entire dataset.
* **Probabilistic Modeling:** Implemented the **Multinomial Naive Bayes** algorithm, which is highly effective for text-based classification.
* **Balanced Evaluation:** Focused on **Precision, Recall, and F1-Score** to ensure that legitimate emails are never accidentally blocked.
* **Model Serialization:** Integrated `joblib` to save the trained pipeline, allowing for instant predictions on new data without retraining.

## 💻 Tech Stack

* **Language:** Python 3.10
* **ML Libraries:** `scikit-learn`, `numpy`
* **Text Processing:** `regex` (re)
* **Data Handling:** `pandas`
* **Model Persistence:** `joblib`

## 📂 Project Structure

```text
├── spamdet.py             # Full NLP pipeline & classifier script
├── spam.csv               # Dataset containing labeled messages
├── spam_model.pkl         # Serialized Naive Bayes model
├── vectorizer.pkl         # Serialized TF-IDF vectorizer
└── README.md              # Project documentation
```

## ⚙️ How to Run

1. **Clone the repo:** `git clone https://github.com/SonamSingh510/spamdet.git`
2. **Install dependencies:** `pip install pandas scikit-learn joblib`
3. **Execute:** `python spamdet.py`
