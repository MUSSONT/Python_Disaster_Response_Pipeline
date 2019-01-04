import nltk
import re
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import string
import pickle
import sys
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
        # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('CleanMessages2', con=engine)

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    y['related']=y['related'].map(lambda x: 1 if x == 2 else x)
    return X, y , y.columns

def compute_text_length(data):
    return np.array([len(text) for text in data]).reshape(-1, 1)

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize,ngram_range=(1, 2), max_features=4000)),
                ('tfidf', TfidfTransformer(use_idf=True))
            ])),          
            ('length', Pipeline([('count', FunctionTransformer(compute_text_length, validate=False))]))])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=4, n_estimators=200)))])
    return pipeline
 




def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_test = Y_test.values
    arr = []
    
    for i,cat in enumerate(category_names):
        barr = []
        barr.append(cat)
        barr.append(accuracy_score(y_test[:, i],y_pred[:,i]))
        barr.append(recall_score(y_test[:, i],y_pred[:,i]))
        barr.append(precision_score(y_test[:, i],y_pred[:,i]))
        barr.append(f1_score(y_test[:, i],y_pred[:,i]))
        arr.append(barr)
        #print(classification_report(y_test[:, i], y_pred[:, i]))
    print(pd.DataFrame(arr, columns=['cat','accuracy','recall','precision','f1_score']))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as fp:
        pickle.dump(model, fp)


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