import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import summarize  # Import the summarize function from summarize.py

# Hugging Face token
HUGGINGFACE_TOKEN = "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"

# Load sentiment model and tokenizer
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, token=HUGGINGFACE_TOKEN)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, token=HUGGINGFACE_TOKEN)

def get_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return "Positive" if predicted_class_id == 1 else "Negative"

st.title("Text Sentiment and Summarizer")

text = st.text_area("Enter a paragraph:", height=200)
if st.button("Analyze"):
    if text:
        sentiment = get_sentiment(text)
        st.write("Sentiment:", sentiment)
        
        summary = summarize.summarize_text(text)  # Call the summarize function from summarize.py
        st.write("Summary:", summary)
