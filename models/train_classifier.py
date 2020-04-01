# import libraries
from sqlalchemy import *
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df =  pd.read_sql_table(database_filepath, engine)
    X = df["message"]
    y = df.iloc[:,4:]
    return X,y,y.keys()


def tokenize(text):
     # Normalize text
        text = text.lower()
        text = re.sub("[^A-Za-z0-9]", ' ', text)
        # Tokenize text
        words = nltk.word_tokenize(text)
        # Remove stop words
        stopWords = set(stopwords.words('english'))
        words = [w for w in words if w not in stopWords]

        return words


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    '''
    parameters = {'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    new_clf=cv.fit(X_train,y_train)
    return new_clf
    '''
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))



def save_model(model, model_filepath):   
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = "data/DisasterResponse.db","models/clf"
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)

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