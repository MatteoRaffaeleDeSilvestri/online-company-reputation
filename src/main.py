from prometheus_client import generate_latest, Counter, Histogram, CONTENT_TYPE_LATEST
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, Response
from pydantic import BaseModel
from torch import softmax
import uvicorn
import torch
import time
import os

# Start app
app = FastAPI()

# Tracks the volume of requests and their success/failure status
PREDICTION_REQUESTS = Counter(
    'model_requests_total',
    'Total count of inference requests',
    ['status'])

# Tracks the latency/processing time for each request
PREDICTION_LATENCY = Histogram(
    'model_request_duration_seconds',
    'Time spent processing the inference request')

# Select model source
MODEL_SOURCE = 'MRDS98/online-company-reputation'

classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)

# Deivce setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)

# Set model to evaluation mode
classifier.eval()

# Data schema for the incoming POST request body
class AnalysisRequest(BaseModel):
    text: str

# Check if the service is running
@app.get('/')
async def health_check():
    return {'message': 'Sentiment Analysis API is running...'}

# Processes input text and returns a list of classification labels with confidence scores
@app.post('/analyze')
async def perform_sentiment_analysis(request_body: AnalysisRequest):

    start_timestamp = time.time()

    try:
        # Tokenize input text for the specific model architecture
        tokenized_inputs = tokenizer(
            request_body.text,
            return_tensors = 'pt',
            truncation = True,
            padding = True
        ).to(device)

        # Perform inference (without calculating gradients)
        with torch.no_grad():
            inference_outputs = classifier(**tokenized_inputs)

        # Extract logits
        raw_logits = inference_outputs.logits

        # Convert probability distribution using Softmax
        class_probabilities = softmax(raw_logits, dim = 1)[0]

        # Map probabilities to readable labels
        classification_results = list()

        for index, probability in enumerate(class_probabilities):
            classification_results.append({
                'label': classifier.config.id2label[index],
                'score': float(probability.item())})

        # Log success metric
        PREDICTION_REQUESTS.labels(status = 'success').inc()

        return classification_results

    # Log failure metric and raise exception
    except Exception as error:
        PREDICTION_REQUESTS.labels(status = 'error').inc()
        print(f'Error during inference: {error}')
        raise

    # Observe total request duration
    finally:
        duration = time.time() - start_timestamp
        PREDICTION_LATENCY.observe(duration)

# Endpoint for Prometheus to scrape monitoring data
@app.get('/metrics')
def export_prometheus_metrics():
    return Response(generate_latest(), media_type = CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    uvicorn.run(app, host = '0.0.0.0', port = 8000)
