# Disaster Response Pipeline Project

## Introduction
The aim of this project is to build a model that is able to predict different categories for given disaster response messages (e.g. Water needed). The project therefore consists of
* an ETL pipeline or cleaning the data and storing it in a database
* an ML pipeline for training and saving the model
* a webapp (using Flask) as the user interface

## Instructions:
You will need two csv files with the raw data (one for the messages and one for the corresponding categories).

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
