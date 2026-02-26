'''
Inference API and monitoring service:
this script implements a REST API for real-time sentiment analysis.
It integrates PyTorch for performance inference, Pydantic for request 
validation and Prometheus instrumentation for observability 
'''

# LIBRARY IMPORT
from prometheus_client import generate_latest, Counter, Histogram, CONTENT_TYPE_LATEST
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, Response
from pydantic import BaseModel
from torch import softmax
import uvicorn
import torch
import time
import os

# SERVICE INITIALIZATION
app = FastAPI()

# OBSERVABILITY CONFIGURATION
PREDICTION_REQUESTS = Counter(
    'model_requests_total',
    'Total count of inference requests',
    ['status'])

PREDICTION_LATENCY = Histogram(
    'model_request_duration_seconds',
    'Time spent processing the inference request')

# MODEL SETUP
MODEL_SOURCE = 'MRDS98/online-company-reputation'

classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)

classifier.eval()

# DATA TRANSFER OBJECTS
class AnalysisRequest(BaseModel):
    
    '''
    Schema definition for incoming API payloads
    '''
    
    text: str

# API ENDPOINT
@app.get('/')
async def health_check():
    
    '''
    Service availability probe
    '''

    return {'message': 'Sentiment Analysis API is running...'}

@app.post('/analyze')
async def perform_sentiment_analysis(request_body: AnalysisRequest):

    '''
    Primary inference route: processes raw text through
    the model pipeline to generate a probability 
    distribution across sentiment classes
    
    Workflow:
        1. Tokenization and sequence alignment;
        2. Forward pass through the Transformer architecture;
        3. Softmax normalization for confidence scoring;
        4. Mapping numerical outputs to human-readable labels
    '''

    start_timestamp = time.time()

    try:
        tokenized_inputs = tokenizer(
            request_body.text,
            return_tensors = 'pt',
            truncation = True,
            padding = True
        ).to(device)

        with torch.no_grad():
            inference_outputs = classifier(**tokenized_inputs)

        raw_logits = inference_outputs.logits

        class_probabilities = softmax(raw_logits, dim = 1)[0]

        classification_results = list()

        for index, probability in enumerate(class_probabilities):
            classification_results.append({
                'label': classifier.config.id2label[index],
                'score': float(probability.item())})

        PREDICTION_REQUESTS.labels(status = 'success').inc()

        return classification_results

    except Exception as error:
        PREDICTION_REQUESTS.labels(status = 'error').inc()
        print(f'Error during inference: {error}')
        raise

    finally:
        duration = time.time() - start_timestamp
        PREDICTION_LATENCY.observe(duration)

@app.get('/metrics')
def export_prometheus_metrics():

    '''
    Endpoint for metrics collection by an external Prometheus instance
    '''

    return Response(generate_latest(), media_type = CONTENT_TYPE_LATEST)

# ENTRY POINT
if __name__ == '__main__':
    uvicorn.run(app, host = '0.0.0.0', port = 8000)
