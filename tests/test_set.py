from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def CI_test_sentiment_analysis():
    
    data = {'text': 'This product works exactly as advertised and has significantly simplified my daily routine'}
    response = client.post('/analize', json = data)

    assert response.status_code == 200
    assert isinstance(response.json(), list)

    for item in response.json():
        assert 'label' in item
        assert 'score' in item
        assert isinstance(item['label'], str)
        assert isinstance(item['score'], float)
