import json
import plotly
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    words = []
    for sent in df['message'].tolist():
        for word in tokenize(sent):
            words.append(word)

    word_counts = pd.Series(words).value_counts()
    top10_word_counts = word_counts[:10]
    top10_words = list(top10_word_counts.index)

    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    top10_category_counts = Y.sum().sort_values(ascending=False)[:10]
    top10_categories = list(top10_category_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },

        {
            'data': [
                Bar(
                    x=top10_words,
                    y=top10_word_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Most Common Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=top10_categories,
                    y=top10_category_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Most Common Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
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
