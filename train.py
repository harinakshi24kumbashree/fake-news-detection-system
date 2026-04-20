import pandas as pd
import re
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# 1️⃣ Load Dataset
# -----------------------
data = pd.read_csv("data/fake_news_dataset.csv")

data["text"] = data["text"].astype(str)
data = data[data["text"].str.len() > 30]

# -----------------------
# 2️⃣ Text Cleaning Function
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]  # 1 = Real, 0 = Fake

# -----------------------
# 3️⃣ TF-IDF Vectorizer
# -----------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1,3),   # unigrams + bigrams + trigrams
    min_df=2
)

X_vec = vectorizer.fit_transform(X)

# -----------------------
# 4️⃣ Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# 5️⃣ Stronger Model (LinearSVC)
# -----------------------
base_model = LinearSVC(class_weight="balanced")

# Calibrate to get probabilities
model = CalibratedClassifierCV(base_model)

model.fit(X_train, y_train)

# -----------------------
# 6️⃣ Evaluation
# -----------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%\n")
print(classification_report(y_test, y_pred))

# -----------------------
# 7️⃣ Save Model
# -----------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model trained and saved successfully.")
