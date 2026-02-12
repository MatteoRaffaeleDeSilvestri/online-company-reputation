# Library import
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from datasets.table import Table
from datasets import Dataset
import pandas as pd
import torch
import json

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
    num_labels = 2,
    ignore_mismatched_sizes = True)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

# Set device (GPU if available, CPU otherwise)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class SentimentAnalysisModel(Dataset):

    def __init__(self, input_encodings, target_labels, arrow_table: Table):
        super().__init__(arrow_table)
        self.encodings = input_encodings
        self.labels = target_labels

    def __getitem__(self, idx):
        # Convert encoding lists to tensors for the current index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Function to process and tokenize the dataset in batches
def process_data_batch(batch):
    return tokenizer(batch['sentence'], truncation=True, max_length=128)

# Main training pipeline
def training_pipeline():

    # Datasets preparation
    combined_dataframe = pd.DataFrame()
    dataset_files = [
        DATASET_DIR + '/amazon_cells_labelled.txt',
        DATASET_DIR + '/imdb_labelled.txt',
        DATASET_DIR + '/yelp_labelled.txt']

    for file_path in dataset_files:
        temp_df = pd.read_csv(file_path, sep='\t', names=['sentence', 'label'])

        # Text pre-processing: convert to lowercase
        temp_df['sentence'] = temp_df['sentence'].str.lower()
        combined_dataframe = pd.concat([combined_dataframe, temp_df])

    # Convert pandas structure to Hugging Face Dataset object
    raw_dataset = Dataset.from_pandas(combined_dataframe)

    # Train - test split
    partitioned_dataset = raw_dataset.train_test_split(test_size=0.2)

    # Configure label mapping for the model
    model.config.id2label = {0: 'negative', 1: 'positive'}
    model.config.label2id = {'negative': 0, 'positive': 1}

    # Text tokenization and padding setup
    tokenized_data = partitioned_dataset.map(process_data_batch, batched=True)
    padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training configuration
    training_configs = TrainingArguments(
        output_dir = CHECKPOINT_DIR,        # Location for model checkpoints
        num_train_epochs = 3,               # Training epochs
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 64,
        warmup_steps = 500,                 # Learning rate scheduler warm-up
        weight_decay = 0.01,                # Regularization to prevent overfitting
        eval_strategy = 'epoch')            # Performance check at the end of each epoch

    # Initialize trainer
    model_trainer = Trainer(
        model = model,
        args = training_configs,
        train_dataset = tokenized_data['train'],
        eval_dataset = tokenized_data['test'],
        data_collator = padding_collator)

    # Execute training
    model_trainer.train()

    # Model evaluation
    model_evaluator = model_trainer.predict(tokenized_data['test'])

    # Extract predictions and ground truth labels
    model_predictions = torch.argmax(torch.tensor(model_evaluator.predictions), dim = 1)
    dataset_labels = model_evaluator.label_ids

    # Model metrics
    model_accuracy = accuracy_score(dataset_labels, model_predictions)
    model_f1_score = f1_score(dataset_labels, model_predictions, average='weighted')

    print(f'Accuracy score: {model_accuracy:.4f}')
    print(f'F1-Score: {model_f1_score:.4f}')

    # Save trained model, tokenizer and performance metrics

    # This creates a folder containing 'model.safetensors' and 'config.json'
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f'Model and tokenizer successfully saved to {MODEL_DIR}')

    metrics_data = {
        'accuracy': float(model_accuracy),
        'f1_score': float(model_f1_score),
        'base_model': HF_MODEL}

    with open(f'{METRICS_FILE}', 'w') as f:
        json.dump(metrics_data, f, indent = 4)
    print(f'Metrics saved successfully to {METRICS_FILE}')

if __name__ == '__main__':
    training_pipeline()
