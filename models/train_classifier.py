"""
Sample Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl

Args:
    1. Path to SQLite db file from previous step
    2. Path to the output ML model
"""

# import libraries
import sys
import os
import re
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download('wordnet')

import numpy as np
import pandas as pd

from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import fbeta_score

def load_data(database_filepath):
    """
    INPUT:
        database_filepath: Path to SQLite db
    Output:
        X: a dataframe containing features
        Y: dataframe containing labels
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    #Remove columns that are all 0 or 1, clean `related` column
    df = df.loc[:,~((df==1).all()|(df==0).all())]
    df['related']=df['related'].map(lambda x: 1 if x == 2 else 0)
    
    X = df['message']
    y = df.iloc[:,4:]

    return X, y


def tokenize(text, url_placeholder_str="urlplaceholder"):
    """
    INPUT:
        text: text that needs to be tokenized
    OUTPUT:
        cleaned_tokens: list of tokens extracted
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_placeholder_str)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    cleaned_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return cleaned_tokens


# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    An estimator trying to extract the first verb of a sentence as a new feature.
    INPUT: 
        BaseEstimator: BaseEstimator class from sklearn.base
        TransformerMixin: TransformerMixin class from sklearn.base
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_pipeline():
    """
    OUTPUT:
        ML Pipeline that processes and classifies message text
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline

def calculate_fscore(y_true,y_pred):
    """
    This functions calculate and output the geometric mean of the fbeta_score on each label.
  
    INPUT:
        y_true: List of label data
        y_prod: List of prediction data
    
    OUTPUT:
        f1score: Geometric mean of fscore
    """
    
    if isinstance(y_pred, pd.DataFrame) == True:
        print('-----------------')
        y_pred = y_pred.values

    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values

    
    score_list = []
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],1,average='weighted')
        score_list.append(score)
        
    f1score = np.asarray(score_list)
    f1score = f1score[f1score<1]
    
    # calculate the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score

def evaluate_pipeline(pipeline, X_test, Y_test):
    """
    This function applies a ML pipeline to the test set and tries to evaluate the pipeline performance
    
    INPUT:
        pipeline: ML Pipeline
        X_test: Test features
        Y_test: Test labels
    """
    Y_pred = pipeline.predict(X_test)
    
    multiple_score = calculate_fscore(Y_test,Y_pred)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Overall accuracy {0:.3f}%'.format(overall_accuracy*100))
    print('F1 score {0:.3f}%'.format(multiple_score*100))

    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance: {}'.format(column))
        print(classification_report(Y_test[column], Y_pred[column]))


def save_model(pipeline, pkl_filepath):
    """
    This function saves trained model as .pkl file.
    
    INPUT:
        pipeline: GridSearchCV or Scikit Pipeline object
        pkl_filepath: destination path to save .pkl file
    
    """
    pickle.dump(pipeline, open(pkl_filepath, 'wb'))

def main():
    """
    1. Extract data from SQLite db
    2. Build ML pipeline with transformers and classifier
    2. Train ML pipeline on training set
    3. Evaluate model performance on test set
    4. Save trained model as .pkl file
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building pipeline ...')
        pipeline = build_pipeline()
        
        print('Training pipeline ...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test)

        print('Saving pipeline \n    MODEL: {}'.format(model_filepath))
        save_model(pipeline, model_filepath)

        print('Trained model saved!')

    else:
          print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()