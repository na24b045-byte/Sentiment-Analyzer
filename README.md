# Sentiment-Analyzer
IMDB Reviews
# Movie Review Sentiment Analyzer

A text classification model that predicts whether a movie review is **positive** or **negative**.

## Tech Stack
- **Model**: Logistic Regression with TF-IDF features (bigrams)
- **Dataset**: IMDB Movie Reviews (50,000 samples)
- **Accuracy**: ~90% on test set
- **Demo**: Gradio web interface

## How It Works
1. Text is vectorized using TF-IDF (Term Frequency–Inverse Document Frequency)
2. Bigrams capture two-word phrases like "not good" or "very bad"
3. Logistic Regression classifies the vector as Positive or Negative

## Run Locally
```bash
pip install -r requirements.txt
python train.py    # trains and saves the model
python app.py      # launches the Gradio demo
```

## Results
| Metric    | Score |
|-----------|-------|
| Accuracy  | ~90%  |
| Precision | ~90%  |
| Recall    | ~90%  |
