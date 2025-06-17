# 🧠 DistilBERT Sentiment Classifier

This is a fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased) model for **sentiment/emotion classification**. It was trained using the Emotion Dataset and built as part of a custom NLP pipeline with Hugging Face Transformers and PyTorch.

---

## 📂 Model Details

- **Base Model**: `distilbert-base-uncased`
- **Task**: Multi-class Text Classification (Sentiment/Emotion)
- **Fine-tuned on**: Emotion Dataset (6 classes)
- **Framework**: PyTorch + Hugging Face Transformers
- **Tokenizer**: `distilbert-base-uncased`

---

## 📊 Labels

The model predicts the following emotions:
- 😄 `happy`
- 😡 `angry`
- 😢 `sad`
- 😱 `fear`
- 😐 `neutral`
- 😍 `love`

---

## 🚀 Example Usage

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("Pranav-1111/sentiment-distilbert")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Predict
text = "I am feeling wonderful today!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
probs = softmax(outputs.logits, dim=1)

# Get label
predicted_class = torch.argmax(probs, dim=1).item()
print(f"Predicted Emotion ID: {predicted_class}")


## 📈 Training Metrics
🧪 Final Accuracy: 98–99% (small subset used)

🔢 Epochs: 10

📉 Loss: gradually decreasing with each epoch

## 🤖 Chatbot UI Demo

This is what the chatbot interface looks like during a real conversation:
![demo](https://github.com/user-attachments/assets/df9b2102-650b-497a-8fce-927f0ace1da1)


## Details
Developed by: Bhatt Pranav
Language(s) (NLP): Python

🤝 Contributing
Pull requests are welcome! If you find a bug or want to improve something, open an issue first.

📜 License
MIT License © 2025 Pranav Bhatt
