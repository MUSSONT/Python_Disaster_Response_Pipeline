import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle 

app = Flask(__name__)


engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CleanMessages2', engine)
print( df.groupby('genre').count()['message'])