import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.svm import SVC
import pickle

def load_data(database_filepath):
    """
    Loads data from sqlite3 database

    Parameters
    ----------
    database_filepath : String
        Path to the sqlite3 database

    Returns
    -------
    X : Pandas.Series
        Message data from the database
    Y : Pandas.DataFrame
        Labels from the database, one column per category
    category_names : List of strings
        List of category names

    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('response_data', engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the given text using lower case, removing punctuation,
    tokenize to single words, lemmatize and stem

    Parameters
    ----------
    text : String
        Text to be tokenized

    Returns
    -------
    stemmed_words : List of strings
        Tokens (stemmed words) extracted from the text

    """
    # lower case
    text = text.lower()
    
    # Remove punctuation
    text = re.sub('\W', ' ', text)
    
    # Word tokenize
    words = nltk.word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if not word in (set(words).intersection(nltk.corpus.stopwords.words('english')))]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_words = []
    for word in words:
        clean_words.append(lemmatizer.lemmatize(word))
    
    # Stemming
    stemmer = SnowballStemmer('english')
    stemmed_words = []
    for word in clean_words:
        stemmed_words.append(stemmer.stem(word))
    
    return stemmed_words


def build_model():
    """
    Build model pipeline of CountVectorizer, TfidfTransformer and
    multiple SVM classifier

    Returns
    -------
    pipeline : sklearn.Pipeline
        Pipeline consisting of transformers and predictor

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SVC()))
    ])
    pipeline.set_params(clf__estimator__class_weight='balanced')
    
    parameters = {'clf__estimator__C': [1.1, 1.25, 1.4], 'clf__estimator__gamma': [0.75, 1, 1.25]}
    cv = GridSearchCV(pipeline, parameters, n_jobs=3)
    #pipeline.set_params(clf__estimator__C=1.25)
    #pipeline.set_params(clf__estimator__gamma=1)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on the test data and prints out the results

    Parameters
    ----------
    model : Model object with .predict() method
        Model (or pipeline) to be evaluated
    X_test : Pandas.Series or Numpy.Array
        Features of the test data
    Y_test : Pandas.DataFrame or Numpy.Array
        Labels corresponding to the test data
    category_names : List of strings
        Names of the labels

    Returns
    -------
    None.

    """
    Y_test_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[category], Y_test_pred[:,i]))


def save_model(model, model_filepath):
    """
    Save the model to the given filepath.

    Parameters
    ----------
    model : sklearn.predictor
        Model to be saved
    model_filepath : string
        Determines where to store the model

    Returns
    -------
    None.

    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
    # if len(sys.argv) == 1:
    #     sys.argv.extend(['../data/DisasterResponse.db', 'classifier.pkl'])
    main()