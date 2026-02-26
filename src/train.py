'''
Sentiment analysis training pipeline:
this file implements a supervised fine-tuning workflow for sequence classification. 
It handles data acquisition, preprocessing and tokenization 
as well as model training and evaluation
'''

# LIBRARY IMPORT
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import pandas as pd
import kagglehub
import torch
import json
import os

# ENVIRONMENT SETUP
MODEL_DIR = './src/sentiment_analysis_model'
CHECKPOINT_DIR = './src/training_checkpoints'
METRICS_FILE = 'metrics.json'

# MODEL INITIALIZATION
HF_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL,
    num_labels = 3)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def dataset_import():

    '''
    Handles dataset acquisition from Kaggle: downloads source data,
    performs label mapping to numerical indices, normalize the text 
    and check data integrity by removing null entries
    
    Returns:
        Dataset: A formatted Hugging Face dataset object
    '''

    df_path = kagglehub.dataset_download('niraliivaghani/flipkart-product-customer-reviews-dataset')

    try:
        df = pd.read_csv(f'{df_path}/{os.listdir(df_path)[0]}', usecols = [4, 5], engine = 'python')
        df.columns = ['sentence', 'label']
        df['sentence'] = df['sentence'].str.lower().str.strip()

        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

        df['label'] = df['label'].map(mapping)
        df = df.dropna(subset=['label'])
        df['sentence'] = df['sentence'].astype(str)
        df = df.dropna(subset=['sentence'])
        df['label'] = df['label'].astype(int)
    except Exception as e:
        print(f'Error loading dataset: {e}')

    return Dataset.from_pandas(df)

def process_data_batch(batch):

    '''
    Applies tokenization and sequence truncation for batch processing
    
    Args:
        batch (dict): A collection of raw text sequences
    Returns:
        dict: Tokenized representation with attention masks
    '''

    return tokenizer(batch['sentence'], truncation = True, max_length = 128)

def training_pipeline():

    '''
    Executes the end-to-end training and evaluation lifecycle
    
    Includes dataset partitioning, model label configuration, 
    application of the hyperparameter and the 
    serialization of the final model and performance metrics
    '''

    # DATA PREPARATION
    raw_dataset = dataset_import()
    partitioned_dataset = raw_dataset.train_test_split(test_size = 0.3)

    model.config.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    model.config.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

    tokenized_data = partitioned_dataset.map(process_data_batch, batched = True)
    padding_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    # TRAINING CONFIGURATION
    training_configs = TrainingArguments(
        output_dir = CHECKPOINT_DIR,
        num_train_epochs = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 64,
        warmup_steps = 500,
        weight_decay = 0.01,
        eval_strategy = 'epoch')

    # TRAINING EXECUTION
    model_trainer = Trainer(
        model = model,
        args = training_configs,
        train_dataset = tokenized_data['train'],
        eval_dataset = tokenized_data['test'],
        data_collator = padding_collator)

    model_trainer.train()

    # PERFORMANCE EVALUATION
    model_evaluator = model_trainer.predict(tokenized_data['test'])

    model_predictions = torch.argmax(torch.tensor(model_evaluator.predictions), dim = 1)
    dataset_labels = model_evaluator.label_ids

    model_accuracy = accuracy_score(dataset_labels, model_predictions)
    model_f1_score = f1_score(dataset_labels, model_predictions, average = 'macro')

    print(f'Accuracy score: {model_accuracy:.4f}')
    print(f'F1-Score: {model_f1_score:.4f}')

    # MODEL PERSISTENCE
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    metrics_data = {
        'accuracy': float(model_accuracy),
        'f1_score': float(model_f1_score),
        'base_model': HF_MODEL}

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_data, f, indent = 4)

if __name__ == '__main__':
    training_pipeline()
