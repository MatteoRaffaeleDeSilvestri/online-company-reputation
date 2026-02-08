from fastapi import FastAPI, HTTPException
from model import SentimentAnalysisModel
from pydantic import BaseModel


app = FastAPI()

sentiment_model = SentimentAnalysisModel()

class TextInput(BaseModel):
    text: str

@app.get('/')
async def root():
    return {'message': 'Sentiment Analysis API acitve'}

@app.get('/status')
async def status():
    return {'status': 'ok'}

@app.post('/predictions')
async def predictions(input: TextInput):
    try:
        predictions = sentiment_model.predict(texts=[input.text])
        return {'predictions': predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))