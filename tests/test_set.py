from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_sentiment_analysis(txt):

    data = {'text': txt}
    response = client.post('/analyze', json=data)

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    
    for item in response.json():
        assert 'label' in item
        assert 'score' in item
        assert isinstance(item['label'], str)
        assert isinstance(item['score'], float)
    
    return response.json()

if __name__ == '__main__':

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
