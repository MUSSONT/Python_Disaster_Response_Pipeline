import json
import plotly
import re
import string
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.externals import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
app = Flask(__name__)
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CleanMessages2', engine)

# load model
with open('../models/classifier.pkl', 'rb') as fp: 
    model = pickle.load(fp)



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category = list(df.columns[4:])
    category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))
    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:11]
    categories_names = list(categories_mean.index) 
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_mean
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()