# Disaster Response Pipeline Project
Repository for second project of Udacity Data Science Nanodegree - building a disaster response pipeline and web app.

## Overview
The goal of the project was to use disaster data from Figure Eight to build a model for an API that classifies new disaster messages.  We were provided with real messages that were sent during disaster events, along with categories matching each message.  We used data cleaning and natural language processing to analyze the text messages, then used the cleaned data to build a machine learning model to classify new messages. <br/><br/>
After creating our model, we exported it as a pickle file and integrated it into a Flask web application, the template of which was provided by Udacity.  The web app allows users to enter new messages and get a classification of what categories the message belongs to.  The web app also provides visualisations giving insights into the original Figure Eight training dataset. <br/><br/>


## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
