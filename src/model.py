from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np
import torch

# Hugging Face model
MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

class SentimentAnalysisModel:

    def __init__(self, device=None):

        # Load tokenizer and pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # Set up device (CPU or GPU)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to the available device
        self.model.to(self.device)
        self.model.eval()
        print(f'Model device: {self.device}')

    def prediction(self, texts):

        # Text tokenization
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        
        # Move input tensors to the device of the model
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        # Performs inference without calculating gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            probs = probabilities.cpu().numpy()
            
        # Select highest probability label index
        labels = np.argmax(probs, axis=1)
        
        # Map the results into a readable format
        label_mapping = ['negative', 
                         'neutral', 
                         'positive']
        results = list()
        
        for l, p in zip(labels, probs):
            results.append({
                'label_id': int(l),
                'label': label_mapping[int(l)],
                'scores': p.tolist()})
            
        return results