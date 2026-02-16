from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

data = {'text': 'This product works exactly as advertised and has significantly simplified my daily routine'}
response = client.post('/analyze', json=data)

assert response.status_code == 200
assert isinstance(response.json(), list)

for item in response.json():
    assert 'label' in item
    assert 'score' in item
    assert isinstance(item['label'], str)
    assert isinstance(item['score'], float)
    
print(f'Sentiment analysis results for input: "{data['text']}"')
for i in sentiment_eval:
    print(f"- {i['label']}: {i['score']:.4f}")
