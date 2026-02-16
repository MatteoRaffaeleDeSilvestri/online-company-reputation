from src.test import test_sentiment_analysis
from fastapi.testclient import TestClient
from src.main import app

if __name__ == '__main__':

    client = TestClient(app)

    # Positive review test
    positive_text = 'This product works exactly as advertised and has significantly simplified my daily routine'
    sentiment_eval = test_sentiment_analysis(positive_text)
    
    print(f'Sentiment analysis results for input: "{positive_text}"')
    for i in sentiment_eval:
        print(f"- {i['label']}: {i['score']:.4f}")

    # Negative review test
    negative_text = 'The instruction manual was incredibly confusing and lacked several crucial steps'
    sentiment_eval = test_sentiment_analysis(negative_text)

    print(f'Sentiment analysis results for input: "{negative_text}"')
    for i in sentiment_eval:
        print(f"- {i['label']}: {i['score']:.4f}")
