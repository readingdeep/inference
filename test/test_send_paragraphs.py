import pandas as pd
import requests


TEST_URL = 'http://192.168.1.182:5000/'


def test_inference():
    df = pd.read_csv('gutenberg_data.csv')
    paragraphs = df['text'].sample(4).to_json(orient='records')
    y_pred = requests.post(TEST_URL, json=paragraphs).json()
    print(y_pred)


if __name__ == '__main__':
    test_inference()