import sys
import re
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(database_filepath):
    """Load the disaster messages dataset from database.

    Args:
    database_filepath: str. the filepath for the disaster messages database, ex:'DisasterResponse.db'

    Returns:
    X: array. the messages to be used as the input feature
    Y: 2d array. the categories' lables to be used as the target variables
    category_names: list. the categories' names 
    """
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine) 

    X = df.message.values    # Select the message column as the input feature
    Y = df.iloc[:,4:].values        # Select all the categories columns as the target variable
    category_names= list(df.columns[4:])     # Extract category names

    return X, Y, category_names

def tokenize(text):
    """Preprocess the input text
    
    This function will apply the following text processing operations:
    
    1-Data Cleaning
    2-Tokinization
    3-Lowercasing
    4-Lemmatization
    5-Stemming
    6-Stop words removal 
    
    Args:
    text: str. the input message to be processed
    
    Returns:
    processed_tokens: list of str. the list of clean extracted tokens
    """
    
    # Create a regex to be used for detecting urls 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect urls and replace it with a unified string 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlstring")

    tokens = word_tokenize(text)   # Split the message into tokens
    
    lemmatizer = WordNetLemmatizer()   # Create a WordNetLemmatizer
    stemmer = PorterStemmer()    # Create a PorterStemmer

    # Lemmatize then Stem the obtained tokens 
    clean_tokens = []
    for tok in tokens:
        lemmatized_token = lemmatizer.lemmatize(tok).lower().strip()
        stemmed_token = stemmer.stem(lemmatized_token)
        clean_tokens.append(stemmed_token)
        
    # Remove the stop words
    processed_tokens = [t for t in clean_tokens if t not in stopwords.words("english")]

    return processed_tokens


def build_model():
    """Build a machine learning pipeline that takes in the `message` column as input
       and output classification results on the 36 categories in the dataset.
    
    Returns:
    cv: classification pipeline
    """
    pipeline = Pipeline([
        ('tfidfvect',TfidfVectorizer(tokenizer=tokenize)),       # Convert a collection of raw documents to a matrix of TF-IDF features
        ('clf',MultiOutputClassifier(RandomForestClassifier()))])       # Random Forest Classifer
    
    
    # Use grid search to find best parameters
    parameters = {
    #    'tfidfvect__ngram_range': [(1,1),(1,2)],
    #    'tfidfvect__min_df': [1,2,3],
        'clf__estimator__n_estimators': [100, 200],
    #    'clf__estimator__max_depth': [None, 5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)    
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model on test dataset and generate classification_report for each output category.
    
    Args:
    model: fitted model
    X_test: array. the input feature of the test dataset
    Y_test: 2d array. the target variables of the test dataset
    category_names: list: the categories' names 

    Returns:
    Print the classification_report for each output category
    """
    
    # Predict the output categories for the test dataset
    y_pred=model.predict(X_test)    
    
    # report the f1 score, precision and recall for each output category
    for i in range(0, Y_test.shape[1]):
        print(category_names[i], classification_report(Y_test[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    """ Save the best model as a pickle file
    
    Args:
    model: fitted model
    model_filepath: str. the filepath of the pickle file to save the model
    
    Returns:
    Saves the best model to a pickle file
    """
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def main():
    """The main function to implement ML Pipeline and save the best model as a pickle file
    """
    
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