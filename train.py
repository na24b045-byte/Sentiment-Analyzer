from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib, os

print("Loading dataset...")
dataset = load_dataset("stanfordnlp/imdb")
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts  = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print(f"Train: {len(train_texts)} samples, Test: {len(test_texts)} samples")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words="english", sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
])

print("Training...")
pipeline.fit(train_texts, train_labels)

preds = pipeline.predict(test_texts)
print(f"\nAccuracy: {accuracy_score(test_labels, preds):.4f}")
print(classification_report(test_labels, preds, target_names=["Negative","Positive"]))

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/sentiment_pipeline.pkl")
print("Model saved!")
