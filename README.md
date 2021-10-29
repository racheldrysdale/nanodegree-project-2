# Disaster Response Pipeline Project
Repository for second project of Udacity Data Science Nanodegree - building a disaster response pipeline and web app.

## Project Overview
The goal of the project was to use disaster data from [Figure Eight (now known as Appen)](https://appen.com/) to build a model for an API that classifies new disaster messages.  We were provided with real messages that were sent during disaster events, along with categories matching each message.  We used data cleaning and natural language processing to analyze the text messages, then used the cleaned data to build a machine learning model to classify new messages. <br/><br/>
After creating our model, we exported it as a pickle file and integrated it into a Flask web application, the template of which was provided by Udacity.  The web app allows users to enter new messages and get a classification of what categories the message belongs to.  The web app also provides visualisations giving insights into the original Figure Eight training dataset.

## Process
1. Started by using Jupyter notebooks to perform some exploratory data analysis and create the ETL and ML pipelines
    - The ETL Pipeline notebook can be found in the "data" directory
    - The ML Pipeline notebook can be found in the "models" directory
2. Re-factored the code and created python scripts to run all the data processing and modelling in the terminal/command prompt
3. Brought the completed model into the Flask web app
4. Added code to perform data manipulation to create visualisations using Plotly
5. Updated HTML code to display new visualisations, making use of Bootstrap

## Other Considerations
- The data in the training set was very imbalanced in that there were no messages or few messages relating to some categories.  In an ideal world the messages would be roughly evenly split between categories.  I tried to implement SMOTE oversampling to help with this but found it difficult and took the decision to move on as I already spent a lot of time on it.
- I tested a range of different machine learning models that support multi-label classification, and chose the decision tree classifier as it gave the highest performance.  The other models I tested were:
    - Extra Tree
    - Ensemble Extra Tree
    - KNeighbors
    - Random Forest
- The final model is not extremely high-performing.  To improve model performance I would have liked to build more text features that could be used and explore more NLP capabilities and modelling methods.  I would also like to explore different combinations of hyper-parameters using GridSearch but this process takes a long time.  I was slightly limited by the time constraints of the Nanodegree but may revisit this project in the future to try and improve the model.
- When building the web application, I wanted to make use of Bootstrap's grid layout to display visualisations side-by-side.  The template used an older version of Bootstrap, so I found the most recent version of Boostrap from their website and modified the template code to work with the utilities in the new version.

## Directory Structure
- app
    - templates
        - go.html - the code that allows the functionality to classify a new message and highlight the related categories
        - master.html - the majority of the code that creates the Flask web app including the landing page and visualisations
    - run.py - script to launch the web app
- data
    - DisasterResponse.db - database file containing cleaned training data
    - ETL Pipeline Preparation.ipynb - notebook containing initial ETL pipeline building
    - disaster_categories.csv - source file containing categories of disaster messages
    - disaster_messages.csv - source file containing disaster messages
    - process_data.py - script to import, join, clean, and reformat data, and save output to a SQLite database
- models
    - ML Pipeline Preparation.ipynb - notebook containing ML model building and testing of different models
    - classifier.pkl - exported final model
    - train_classifier.py - script to build and apply machine learning pipeline, and export final model as a pickle file
- readme.md

## Instructions
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Installation and Packages
The following packages were used to complete the Disaster Response Pipeline project:
- sys
- pandas
- numpy
- sklearn
- pickle
- plotly
- json
- re
- nltk
- flask
- matplotlib
- joblib
- sqlalchemy

In the NLTK library, the 'punkt', 'stopwords', and 'wordnet' packages were used.

## Acknowledgements
The data was provided by [Figure Eight (now known as Appen)](https://appen.com/).  The project and templates were provided by [Udacity](https://www.udacity.com/) as part of the Data Science Nanodegree.  In completing the project I undertook a lot of online research and particularly took advantage of the [scikit-learn](https://scikit-learn.org/stable/) and [Bootstrap](https://getbootstrap.com/) documentation, as well as the Udacity lessons and mentor knowledge forum.
