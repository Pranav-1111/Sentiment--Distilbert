from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch

# Load your fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained(
    "C:/Users/Pranav/Desktop/Emotion - Model/sentiment-model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "C:/Users/Pranav/Desktop/Emotion - Model/sentiment-model"
)
# Set to eval mode
model.eval()

# Label map
labels = ['positive', 'negative', 'neutral']

# Input sentence
text = "I love working on AI projects!"

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    predicted = torch.argmax(outputs.logits)

print(f"🧠 Input: {text}")
print(f"🎯 Predicted Label: {labels[predicted]}")
