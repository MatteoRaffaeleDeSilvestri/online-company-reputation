# Library import
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import pandas as pd
import torch
import json
import os

# DIRECTORIES & FILES
DATASET_DIR = './datasets'
MODEL_DIR = './src/sentiment_analysis_model'
CHECKPOINT_DIR = './src/training_checkpoints'
METRICS_FILE = 'metrics.json'

# Hugging Face model
HF_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

# Initialize model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL,
    num_labels = 3)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to process and tokenize the dataset in batches
def process_data_batch(batch):
    return tokenizer(batch['sentence'], truncation = True, max_length = 128)

# Main training pipeline
def training_pipeline():

    # Datasets preparation
    df = pd.read_csv(f'{DATASET_DIR}/Dataset-SA.csv', usecols = [3, 5], engine='python')
    df.columns = ['sentence', 'label']
    df['sentence'] = df['sentence'].str.lower().str.strip()

    mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    df['label'] = df['label'].map(mapping)
    df = df.dropna(subset=['label'])
    df['sentence'] = df['sentence'].astype(str)
    df = df.dropna(subset=['sentence'])
    df['label'] = df['label'].astype(int)

    raw_dataset = Dataset.from_pandas(df)
    partitioned_dataset = raw_dataset.train_test_split(test_size = 0.3)

    model.config.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    model.config.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Text tokenization and padding setup
    tokenized_data = partitioned_dataset.map(process_data_batch, batched = True)
    padding_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    # Training configuration
    training_configs = TrainingArguments(
        output_dir = CHECKPOINT_DIR,
        num_train_epochs = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 64,
        warmup_steps = 500,
        weight_decay = 0.01,
        eval_strategy = 'epoch'
    )

    # Initialize trainer
    model_trainer = Trainer(
        model = model,
        args = training_configs,
        train_dataset = tokenized_data['train'],
        eval_dataset = tokenized_data['test'],
        data_collator = padding_collator)

    # Execute training and evaluation
    model_trainer.train()
    model_evaluator = model_trainer.predict(tokenized_data['test'])

    # Extract predictions and ground truth labels
    model_predictions = torch.argmax(torch.tensor(model_evaluator.predictions), dim = 1)
    dataset_labels = model_evaluator.label_ids

    # Model metrics
    model_accuracy = accuracy_score(dataset_labels, model_predictions)
    model_f1_score = f1_score(dataset_labels, model_predictions, average = 'macro')

    print(f'Accuracy score: {model_accuracy:.4f}')
    print(f'F1-Score: {model_f1_score:.4f}')

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Save trained model, tokenizer and performance metrics
    metrics_data = {
        'accuracy': float(model_accuracy),
        'f1_score': float(model_f1_score),
        'base_model': HF_MODEL}

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_data, f, indent = 4)

if __name__ == '__main__':
    training_pipeline()
