from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk  # only needed if you save/load dataset (optional)
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset

# === 1. Load and preprocess data (same as before)
df = pd.read_csv("Emotion - Model/sentiment_sample.csv")
label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df['label'] = df['label'].map(label_map)

dataset = Dataset.from_pandas(df)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# === 2. Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# === 3. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    logging_steps=1,
    logging_dir="./logs",
    save_strategy="no"
)

# === 4. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# === 5. Train!
trainer.train()

# === 6. Save model
model.save_pretrained("sentiment-model")
tokenizer.save_pretrained("sentiment-model")

print("✅ Model fine-tuned and saved to 'sentiment-model/'")
