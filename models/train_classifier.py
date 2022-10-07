import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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
    '''
    This function loads the data from sqlite database file.

    After the data is loaded into the dataframe, it is divided into X and Y dataframes to later be used in the machinelearning process.
    Messages are stored in "X" and the allocated categories are stored in "Y". 
    The id, original message, and genre are not saved nor in X neither in Y, as they are not needed.

    Parameters:
        database_filepath: This is the filepath to the messages csv file which contains the messages.

    Returns:
        X: The text of the messages.
        Y: The categories allocated to the messages.
        Y.columns.values: Name of the categories that each of the texts can be allocated to.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    
    X = df.message
    Y = df.iloc[:,4:]
    return X, Y, Y.columns.values


def tokenize(text):
    '''
    This function tokenizes an input text.

    First, urls are replaced by "urlplaceholder". Then, the text is tokenized into the words and english stop words are removed.
    Lastly, these words are lemmatized, normalized, stripped, and added to the list of clean tokens.

    Parameters:
        text: The input raw text to be tokenized.

    Returns:
        clean_tokens: The clean tokenized list of normalized words from the input text.
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    This function builds a machine learning model using pipeline and tunes it with the help of gridsearch.

    First a pipeline of CountVectorizer, TF-IDF transformer, and SGD Classifier is initiated. Then, some parameters are given to the 
    grid search, so this pipeline is fine tuned to reach the best results. Finally, best estimator of this gridsearch is returned as
    the output of the function.

    Parameters:
        no input parameters.

    Returns:
        model: a tuned model and the best estimator of the Grid search results.

    After lots of trial with different classification algorithms, I found out the SGDClassifier() has the best performance, 
    considering time, therefore instead of the randomforest that we saw in the class, I will use SGDClassifier() for my pipleline. 
    "ML Pipeline Preparation.ipynb" can be referred for further details about how different algorithms and features in grid search have performed
    '''
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(SGDClassifier()))])

    print("\n Pipeline initiated")

    parameters = {
        'vect__tokenizer':[tokenize, None],
        'vect__max_df': [0.1, 0.5, 1.0],
        'tfidf__use_idf': [True, False],
        'clf__estimator__max_iter': [None, 20],
        'clf__estimator__penalty': ['l2', 'elasticnet']
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)
    cv.fit(X_train, y_train)
    print("\nGridSearchCV started. It might take some time depending on your environment performance")
    print("\n*******************************************")
    print("\nBest Parameters:", cv.best_params_)
    print("\n*******************************************")

    model = cv.best_estimator_

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        This function as input receives already trained and fitted model, test data and category names.
        It prints the Precision, Recall, and F1-Score for each category and also the overall value.
    '''

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
    '''
    This function exports the model as a pickle file and saves it for later.

    Parameters:
        model: The model to be saved.
        model_filepath: The filepath which the model will be saved to.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    '''
    This is the main function. It does not have any input parameters.

    First it checks if all the required arguments are given by the user on the system. It uses "database_filepath" to load the data and
    then trains the model on it. After evaluating the model and printing its metrics, it saves the model into the "model_filepath" and 
    prints the success message.
    If not all the inputs are provided, it informs the user.
    '''
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