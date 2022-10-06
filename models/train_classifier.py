import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    
    # As explained in the notebook, two columns need some attention.
    # "child_alone" column is all zeros and I will drop it.
    # there are 188 2s in "related" column. I will replace them with 1.
    
    df.drop("child_alone", axis=1, inplace=True)
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    X = df.message
    Y = df.iloc[:,4:]
    return X, Y, Y.columns.values


def tokenize(text):
    # As shown in the lectures, I will drop URLs and put "urlplaceholder" instead
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Here first the text will be tokenized into words and then a lemmetizer will be initiated
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Each tokenized word is lemmatized, normalized, stripped, and added to the list of all clean tokens.
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # After lots of trial with different classification algorithms, I found out the SGDClassifier()
    # has the best performance, considering time, therefore instead of the randomforest that we saw in the class,
    # I will use SGDClassifier() for my pipleline. "ML Pipeline Preparation.ipynb" can be referred for further details about
    # how different algorithms and features in grid search have performed
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(SGDClassifier()))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    precision = []
    recall = []
    f1_score = []
    for i, category in enumerate(category_names):
        print("\nFor \"{}\" we have the following values for precision, recall, f1-score: ".format(category))
        print(classification_report(Y_test[category], Y_pred[:,i]).split()[-4:-1])
        print("-----------------------------------------------------------------------------")
        precision.append(float(classification_report(Y_test[category], Y_pred[:,i]).split()[-4]))
        recall.append(float(classification_report(Y_test[category], Y_pred[:,i]).split()[-3]))
        f1_score.append(float(classification_report(Y_test[category], Y_pred[:,i]).split()[-2]))
    print("\n***************************************")
    print("Overall precision is: {:.2f}, recall is: {:.2f}, and f1-score is: {:.2f}".format(np.array(precision).mean(),
                                                                                            np.array(recall).mean(),
                                                                                            np.array(f1_score).mean()))
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()