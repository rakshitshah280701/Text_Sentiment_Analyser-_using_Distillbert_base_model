from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Hugging Face token
HUGGINGFACE_TOKEN = "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"

# Load summarization model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = T5ForConditionalGeneration.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

def summarize_text(text):
    input_text = f"summarize: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
