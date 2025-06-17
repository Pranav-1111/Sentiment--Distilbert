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

labels = ['positive 😊', 'negative 😢', 'neutral 😐']

st.set_page_config(page_title="EmotiBot 💬", page_icon="🤖")
st.title("🤖 EmotiBot – Custom Sentiment Chatbot")
st.markdown("Type anything below and see what the bot thinks!")

user_input = st.chat_input("How are you feeling today?")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits)

    reply = f"🧠 I think you're feeling **{labels[prediction]}**"
    with st.chat_message("assistant"):
        st.write(reply)
