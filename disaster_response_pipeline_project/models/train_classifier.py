import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk import word_tokenize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Loads dataframe from SQLite database and splits into X and Y variables for Machine Learning.
    
    INPUT:
    database_filepath - (str) filepath of the database
    
    OUTPUT:
    X - (pandas dataframe) dataframe containing typed messages
    Y - (pandas dataframe) dataframe containing category flags
    category_names - (list) list of category names
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)

    # Features
    X = df['message']

    # Targets
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    # Category names
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Takes text strings and performs cleaning, tokenizing, lemmatizing, and stopword removal.
    
    INPUT:
    text - (str) text string to be cleaned and tokenized
    
    OUTPUT:
    tokens - (list) list of text tokens
    '''

    # normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stopwords
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words("english")]

    return tokens


def build_model():
    '''
    Creates model pipeline and uses GridSearch to find the best hyper-parameters.
    
    OUTPUT:
    model - (DecisionTreeClassifier) trained multi-output decision tree classification model
    '''

    # Create model pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])

    # Providing parameters for GridSearch
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__max_depth': [10, 20],
        'clf__estimator__min_samples_split': [2, 5],
        'clf__estimator__ccp_alpha': [0.0, 0.1]
    }

    # Build model using GridSearch
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model on the test data and outputs the precision, recall, and F1-score of the overall
    model and each individual category.
    
    INPUT:
    model - (DecisionTreeClassifier) trained multi-output decision tree classification model
    X_test - (df) text messages (features) from the test set
    Y_test - (df) category flags (targets) from the test set
    category_names - (list) list of category names
    '''

    # Create predictions
    Y_pred = model.predict(X_test)

    # Turn predictions into a dataframe
    Y_pred_df = pd.DataFrame(Y_pred, index=Y_test.index, columns=category_names)

    # Print classification report
    print(classification_report(Y_test, Y_pred_df, target_names=category_names, zero_division=0))


def save_model(model, model_filepath):
    '''
    Saves the trained model as a pickle file.
    
    INPUT:
    model - (DecisionTreeClassifier) trained multi-output decision tree classification model
    model_filepath - (str) filepath the model is to be saved to
    '''

    # Save model as a pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    '''
    Script to execute training, evaluation, and saving of the model.
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
