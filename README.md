# Disaster Response Pipeline Project

## Project Description
This project is part of Udacity's Data Scientist Nanodegree Program, it aims to build a machine learning pipeline to classify disaster messages. The dataset that will be used for this project was provided by [Figure Eight](https://www.figure-eight.com/) and contains real messages that were sent during disaster events. We’ll build a machine learning pipeline to categorize these events so that the messages can be sent to the appropriate disaster relief agency.

Finally, the project will include a web app that can be used to get the classification results for an input message and display visualizations of the data.

## Installation
Python 3.6 or above should be used, you’ll need to install the following libraries:

1.	numpy
2.	pandas
3.	sklearn
4.	sqlalchemy
5.	flask
6.	plotly
7.	pickle
8.	sys
9.	re

## File Descriptions

app<br>
| - template<br>
| |- master.html  # main page of web app<br>
| |- go.html  # classification result page of web app<br>
|- run.py  # Flask file that runs app<br>

- data<br>
|- disaster_categories.csv  # data to process<br>
|- disaster_messages.csv  # data to process<br>
|- process_data.py # ETL Pipeline script<br>
|- DisasterResponse.db   # database to save clean data to<br>

- models<br>
|- train_classifier.py # ML Pipeline script<br>
|- classifier.pkl  # saved model<br>

- README.md<br>

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots of the Web App

![Screenshot 1]()
![Screenshot 2]()
![Screenshot 3]()

## Acknowledgements

1. Thanks to [Udacity](www.udacity.com) for this Data Scientist Nanodegree Program.
2. Thanks to [Figure Eight](https://www.figure-eight.com/) for providing the dataset for this project.

