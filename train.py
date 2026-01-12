import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/messages.csv")

X = df["text"]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2
    )),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)

print("Classification report:\n")
print(classification_report(y_val, y_pred))

macro_f1 = f1_score(y_val, y_pred, average="macro")

print("Macro F1:", round(macro_f1, 3))

os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline.named_steps["tfidf"],
    "models/vectorizer.joblib"
)

joblib.dump(
    pipeline.named_steps["model"],
    "models/model.joblib"
)

print("Model and Vectorizer dumped successfully")
