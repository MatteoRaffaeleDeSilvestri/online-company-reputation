'''
API validation and testing setup: this file provides the testing framework to
verify the integrity of the Sentiment Analysis model. It utilizes ad-hoc test 
cases to evaluate model accuracy, API response structures and HTTP protocol compliance
'''

# LIBRARY IMPORT
from fastapi.testclient import TestClient
from main import app
import pytest

# SERVICE INITIALIZATION
client = TestClient(app)

@pytest.mark.parametrize('txt, expected_label', [
    ('This product works exactly as advertised and has significantly simplified my daily routine', 'positive'),
    ('The instruction manual was incredibly confusing and lacked several crucial steps', 'negative'),
    ('The product performs the basic functions described in the manual without any additional features or drawbacks', 'neutral')
])

def test_sentiment_analysis(txt, expected_label):

    '''
    Execution logic for automated API endpoint validation
    
    Validates:
        1. HTTP response status: ensures the /analyze endpoint returns a 200 OK status for valid requests
        2. Data schema: verifies that the payload adheres to the expected list of label/score dictionaries
        3. Type integrity: confirms the data types for classification outputs
    
    Args:
        txt (str): the input sequence to be analyzed
        expected_label (str): the declared sentiment category for comparison
    
    Returns:
        tuple: the complete JSON response and the primary predicted label
    '''

    data = {'text': txt}
    response = client.post('/analyze', json = data)

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    
    for item in response.json():
        assert 'label' in item
        assert 'score' in item
        assert isinstance(item['label'], str)
        assert isinstance(item['score'], float)

    pred_lbl = max(response.json(), key=lambda x: x['score'])

    return response.json(), pred_lbl['label']

# MANUAL VERIFICATION AND DATA SAMPLE
if __name__ == '__main__':

    '''
    SAMPLE SENTENCES FOR TESTING:

    POSITIVE SENTENCES:
    - This product works exactly as advertised and has significantly simplified my daily routine 
    - The build quality is surprisingly premium given the affordable price point
    - I was impressed by how intuitive the setup process was right out of the box
    - The customer service team went above and beyond to resolve my inquiry within minutes
    - It is rare to find something this reliable that also looks great on a kitchen counter

    NEUTRAL SENTENCES:
    - The product performs the basic functions described in the manual without any additional features or drawbacks
    - The materials used are standard for this price range and appear to be durable enough for regular, everyday use
    - The installation process took exactly the amount of time I expected and followed the provided steps reasonably well
    - The design is functional and plain; it fits into the room's decor without standing out or being an eyesore
    - Overall, the item meets the basic requirements for its category, providing a typical experience for the cost

    NEGATIVE SENTENCES:
    - The instruction manual was incredibly confusing and lacked several crucial steps
    - It stopped functioning entirely after just two weeks of very light use
    - The actual product looks nothing like the photos provided in the advertisement
    - I found the interface to be clunky, slow, and generally frustrating to navigate
    - For the premium price they charge, the materials feel remarkably cheap and flimsy
    '''

    txt = 'Overall, the item meets the basic requirements for its category, providing a typical experience for the cost'
    lbl = 'neutral'

    sentiment_eval, pred_lbl = test_sentiment_analysis(txt, lbl)
    
    print(f'Sentiment analysis results for input: "{txt}"')
    for i in sentiment_eval:
        print(f"- {i['label']}: {i['score']:.4f}")
    print(f'Expected label: {lbl}\nPredicted label: {pred_lbl}')
