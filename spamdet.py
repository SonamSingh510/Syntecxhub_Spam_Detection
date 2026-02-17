import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Load Dataset
# Assuming a standard 'spam.csv' with columns: 'v1' (label) and 'v2' (text)
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
except FileNotFoundError:
    print("Please ensure 'spam.csv' is in your Syntecxhub folder.")

# 2. Preprocess Text (Tokenize & Clean)
def clean_text(text):
    text = text.lower() # Lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    return text

df['text'] = df['text'].apply(clean_text)

# 3. Vectorization (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])
y = df['label']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluation (Precision, Recall, F1)
y_pred = model.predict(X_test)
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 7. Save the Pipeline (Vectorizer + Model)
joblib.dump(tfidf, 'vectorizer.pkl')
joblib.dump(model, 'spam_model.pkl')
print("\n✅ Pipeline saved for future predictions.")

# 8. Test Script
def check_spam(message):
    cleaned = clean_text(message)
    vectorized = tfidf.transform([cleaned])
    result = model.predict(vectorized)
    return result[0]

# Example usage:
test_msg = "WINNER! You have won a 1000 dollar prize. Call now to claim!"
print(f"\nTest Message: '{test_msg}'")
print(f"Prediction: {check_spam(test_msg)}")