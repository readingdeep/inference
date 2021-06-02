import json
import os
import pickle

import nltk
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS, cross_origin
import flask

LABELS = "model/labeled_words.pkl"
COUNT_VECTORIZER = "model/count_vectorizer.pkl"
VOCAB = "model/vocab.pkl"
NEWLINE = '\n'
EMPTY = ''
REMOVE_PUNCTUATION = r'[^\w\s]'
TEXT_IDX = -1
WHITESPACE = " "
NUM_LABELS = 10
ARRAY_INSIDE_ARRAY_IDX = 0
LABEL_IDX = 1

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
nltk.download('punkt')

with open(LABELS, 'rb') as pkl_file:
    labeled_words = pickle.load(pkl_file)
with open(COUNT_VECTORIZER, 'rb') as pkl_file:
    cv = pickle.load(pkl_file)
with open(VOCAB, 'rb') as pkl_file:
    vocab = pickle.load(pkl_file)


def clean_data(data):
    """
    TODO
    """
    df = data.copy()
    df = df.iloc[:, TEXT_IDX].str.replace(NEWLINE, EMPTY)
    df = df.str.replace(REMOVE_PUNCTUATION, EMPTY, regex=True).str.lower()
    df = df.map(nltk.word_tokenize)
    return df


def replace_if_not_in_vocab(lst_token, vocab):
    result = []
    for t in lst_token:
        try:
            idx = vocab.index(t)
            result.append(t)
        except ValueError:
            result.append('unk')
    return result


@app.route('/', methods=['POST'])
def predict():
    # def predict(label, cv, vocab, paragraphs):
    params = flask.json.loads(request.get_json())
    # params = json.loads(paragraphs)
    X = pd.DataFrame(params)
    clean_X = clean_data(X)
    y_pred = np.zeros(clean_X.shape[0]).astype(str)
    for i, row in clean_X.items():
        paragraph = WHITESPACE.join(replace_if_not_in_vocab(row, vocab))
        X_cv = cv.transform([paragraph]).toarray()
        counter = labeled_words.iloc[:, :NUM_LABELS].values.T @ X_cv[ARRAY_INSIDE_ARRAY_IDX]
        y_pred[i] = labeled_words.columns[np.argmax(counter)][LABEL_IDX]
    response = y_pred.tolist()
    return f"{json.dumps(response)}"


def main():
    """
    """

    with open(LABELS, 'rb') as pkl_file:
        labeled_words = pickle.load(pkl_file)
    with open(COUNT_VECTORIZER, 'rb') as pkl_file:
        cv = pickle.load(pkl_file)
    with open(VOCAB, 'rb') as pkl_file:
        vocab = pickle.load(pkl_file)

    # Run flask
    try:
        port = int(os.environ.get('PORT'))
        # Running on Heroku
        if port:
            app.run(host='0.0.0.0', port=int(os.environ.get('PORT')))
    # Running locally
    except TypeError:
        app.run(host='0.0.0.0')


if __name__ == '__main__':
    # with open(LABELS, 'rb') as pkl_file:
    #     labeled_words = pickle.load(pkl_file)
    # with open(COUNT_VECTORIZER, 'rb') as pkl_file:
    #     cv = pickle.load(pkl_file)
    # with open(VOCAB, 'rb') as pkl_file:
    #     vocab = pickle.load(pkl_file)
    # df = pd.read_csv('/Users/leonardorosenberg/repos/itc-hackathon/inference/test/gutenberg_data.csv')
    # paragraphs = df['text'].sample(4).to_json(orient='records')
    # y_pred = predict(labeled_words, cv, vocab, paragraphs)
    # print(y_pred)
    main()
