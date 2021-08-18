import json
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, word_tokenize
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask, abort
from flask import request
from flask_cors import CORS
from sqlalchemy import create_engine

app = Flask(__name__)
CORS(app, supports_credentials=True)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data & model
engine = create_engine('sqlite:///../data/disaster_response_db.db')
df = pd.read_sql_table('df', engine)
model = joblib.load("../models/classifier.pkl")


@app.route('/message/message_stat/v1', methods=["POST"])
def message_stat():
    if request.method == 'POST':
        try:
            genre_counts = df.groupby('genre').count()['message']
            genre_names = list(genre_counts.index)

            # format genre stat result
            genre_result = []
            for name in genre_names:
                genre_result.append({
                    'name': name,
                    'value': int(genre_counts[name])
                })

            # format cateogry stat result
            category_names = df.iloc[:, 4:].columns
            category_boolean = (df.iloc[:, 4:] != 0).sum()

            category_result = []
            for name in category_names:
                category_result.append({
                    'name': name,
                    'value': int(category_boolean[name])
                })

            return {
                'genre_distribution': genre_result,
                'category_count': category_result
            }
        except:
            # invalid input parameter
            abort(400)
    else:
        # unsupported method
        abort(405)


@app.route('/message/message_classification/v1', methods=["POST"])
def message_classification():
    if request.method == 'POST':
        try:
            body = json.loads(request.data.decode('utf-8'))
            # classification for input text
            classification_labels = model.predict([body['text']])[0]
            classification_results = dict(
                zip(df.columns[4:], classification_labels.astype(str)))
            return classification_results
        except:
            # invalid input parameter
            abort(400)
    else:
        # unsupported method
        abort(405)


def main():
    app.run(host='localhost', port=3000, debug=True)


if __name__ == '__main__':
    main()
    print('----- Web page needs to be launched in dashboard folder in port 8080 -----')
    print('----- api schema can be found in api-doc.yaml -----')
