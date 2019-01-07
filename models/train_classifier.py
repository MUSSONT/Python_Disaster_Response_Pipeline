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
        """func to get data from db and prep it for use
        args:
        database_filepath - locatuon of db
        returns - 
        X - features
        y - target variable
        & column names for later use """
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('CleanMessages2', con=engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    y['related']=y['related'].map(lambda x: 1 if x == 2 else x)
    return X, y , y.columns

def compute_text_length(data):
    """works out text length of each string
    args: data = message
    outputs: length of message"""
    return np.array([len(text) for text in data]).reshape(-1, 1)

def tokenize(text):
    """func to take mesages and apply token trans, remove web links , punctuation, stop words
    args - message text
    outputs : cleaned tokenized text"""
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
    """func that creates and returns pipeline"""
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
    """func to provide statistics for model evaluation
    args:
    model - the trained pipeline
    X_test - features for testing
    Y_test - target variables for testing
    category_names - colummnn names for mapping
    
    outputs - prints stats for comparing performance of the model"""
    
    y_pred = model.predict(X_test)
    y_test = Y_test.values
    allstats_array = []
    for i,cat in enumerate(category_names):
        catstats_array = []
        catstats_array.append(cat)
        catstats_array.append(accuracy_score(y_test[:, i],y_pred[:,i]))
        catstats_array.append(recall_score(y_test[:, i],y_pred[:,i]))
        catstats_array.append(precision_score(y_test[:, i],y_pred[:,i]))
        catstats_array.append(f1_score(y_test[:, i],y_pred[:,i]))
        allstats_array.append(catstats_array)
        #print(classification_report(y_test[:, i], y_pred[:, i]))
    print(pd.DataFrame(allstats_array, columns=['cat','accuracy','recall','precision','f1_score']))

def save_model(model, model_filepath):
    """ func to save model as pickle
    
    args :
    model - trained pipeline
    mdel_filepath - name and destination to save"""
    
    with open(model_filepath, 'wb') as fp:
        pickle.dump(model, fp)


def main():
    """func to cycle through all the above funx and print progress"""
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