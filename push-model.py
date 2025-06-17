from transformers import AutoTokenizer, DistilBertForSequenceClassification

# Load your fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained(
    "C:/Users/Pranav/Desktop/Emotion - Model/sentiment-model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "C:/Users/Pranav/Desktop/Emotion - Model/sentiment-model"
)

# Replace "your-username/your-model-name" below 👇
model.push_to_hub("Pranav-1111/sentiment-distilbert")
tokenizer.push_to_hub("Pranav-1111/sentiment-distilbert")
