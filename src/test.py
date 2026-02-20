from fastapi.testclient import TestClient
from main import app
import pytest

# Start app
client = TestClient(app)

@pytest.mark.parametrize('txt', [
    'This product works exactly as advertised and has significantly simplified my daily routine',
    'The instruction manual was incredibly confusing and lacked several crucial steps',
    'I was impressed by how intuitive the setup process was right out of the box',
    'For the premium price they charge, the materials feel remarkably cheap and flimsy'])

# Test function for sentiment analysis
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

    '''
    SAMPLE SENTENCES FOR TESTING:

    POSITIVE SENTENCES:
    - This product works exactly as advertised and has significantly simplified my daily routine 
    - The build quality is surprisingly premium given the affordable price point
    - I was impressed by how intuitive the setup process was right out of the box
    - The customer service team went above and beyond to resolve my inquiry within minutes
    - It is rare to find something this reliable that also looks great on a kitchen counter

    NEGATIVE SENTENCES:
    - The instruction manual was incredibly confusing and lacked several crucial steps
    - It stopped functioning entirely after just two weeks of very light use
    - The actual product looks nothing like the photos provided in the advertisement
    - I found the interface to be clunky, slow, and generally frustrating to navigate
    - For the premium price they charge, the materials feel remarkably cheap and flimsy
    '''

    txt = 'This product works exactly as advertised and has significantly simplified my daily routine'
    sentiment_eval = test_sentiment_analysis(txt)
    
    print(f'Sentiment analysis results for input: "{txt}"')
    for i in sentiment_eval:
        print(f"- {i['label']}: {i['score']:.4f}")
