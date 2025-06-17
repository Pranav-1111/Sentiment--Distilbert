import streamlit as st
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch

# Load your fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained(
    "C:/Users/Pranav/Desktop/Emotion - Model/sentiment-model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "C:/Users/Pranav/Desktop/Emotion - Model/sentiment-model"
)
model.eval()

labels = ['positive', 'negative', 'neutral']

st.title("🧠 Custom Sentiment Classifier")
st.write("Enter a sentence and see what your model predicts:")

user_input = st.text_input("Your sentence here:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits)

    st.success(f"🎯 Prediction: **{labels[prediction]}**")







