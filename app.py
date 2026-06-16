import gradio as gr
import joblib

pipeline = joblib.load("model/sentiment_pipeline.pkl")

def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text."
    
    pred = pipeline.predict([text])[0]
    prob = pipeline.predict_proba([text])[0]
    
    label = "Positive 😊" if pred == 1 else "Negative 😞"
    confidence = prob[pred] * 100
    
    return f"{label}  (confidence: {confidence:.1f}%)"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Paste a movie review here...",
        label="Review Text"
    ),
    outputs=gr.Textbox(label="Sentiment"),
    title="Movie Review Sentiment Analyzer",
    description="Trained on 50,000 IMDB reviews using TF-IDF + Logistic Regression. Achieves ~90% accuracy.",
    examples=[
        ["This movie was absolutely fantastic! The acting was superb and the plot kept me hooked."],
        ["Terrible film. Boring, predictable, and a complete waste of time."],
        ["It was okay. Some good moments but overall pretty average."]
    ]
)

if __name__ == "__main__":
    demo.launch()
