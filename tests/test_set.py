from fastapi.testclient import TestClient
from src.main import app

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
    positive_txt = 'This product works exactly as advertised and has significantly simplified my daily routine'    
    positive_eval = test_sentiment_analysis(positive_txt)
    
    print(f'Sentiment analysis results for input: "{positive_txt}"')
    for i in positive_eval:
        print(f"- {i['label']}: {i['score']:.4f}")

    # Negative review test
    negative_txt = 'The instruction manual was incredibly confusing and lacked several crucial steps'
    negative_eval = test_sentiment_analysis(negative_txt)

    print(f'Sentiment analysis results for input: "{negative_txt}"')
    for i in negative_eval:
        print(f"- {i['label']}: {i['score']:.4f}")
