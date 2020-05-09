import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('response_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # data for distribution of message lengths
    y_raw, x_raw = np.histogram(df['message'].apply(len), bins=1000)
    n = 50
    y_len = y_raw[:n]
    x_len = []
    for i in range(n):
        mid = int((x_raw[i]+x_raw[i+1])/2)
        x_len.append(mid)
    
    # data for distribution of number of categories
    Y = df.drop(columns=['id','message','original','genre'])
    y_raw, x_raw = np.histogram(Y.sum(axis=1), bins=30)
    y_cat = y_raw
    x_cat = []
    for i in range(len(y_cat)):
        x_cat.append(int((x_raw[i]+x_raw[i+1])/2))
    
    # data for correlation heatmap
    z = list(np.array(Y.corr()))
    x_cor = Y.columns
    y_cor = Y.columns
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=x_len,
                    y=y_len
                )
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Length"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_cat,
                    y=y_cat)],
            'layout': {
                'title': 'Distribution of active Categories',
                'yaxis': {
                    'title': "Count"},
                'xaxis': {
                    'title': "Active Categories"}
                }
        },
        {
            'data': [
                Heatmap(
                    z=z, x=x_cor, y=y_cor)],
            'layout': {
                'title': 'Correlations between Categories',
                'yaxis': {
                    'title': "Categories"},
                'xaxis': {
                    'title': "Categories"}
                }
        },
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