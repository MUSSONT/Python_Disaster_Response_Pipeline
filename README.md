# Disaster Response Pipeline Project

## Project Summary
Datasets containing real messages that were sent during disaster events. 
Project aim is to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency. E.g. If the message relates to a shortage of food it could be automatically forwarded to the charity/org that is dealing with that.

## Files description
Disaster_messages.csv - contains message id and text of message
Disaster_categories.csv - contains message id and string of categories with binary indicator
process_data.py - performs ET and Loads into sqllite disasterresponse.db
train_classifier.py - trains model based on data from disasterresponse.db and exports model as pkl file
run.py - kicks off flask process

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## future dev
- [PorterStemmer().stem(w) for w in words if w not in stopwords.words("english")].
- build custom transformers to extract whether a message includes a question, specific words, locations, or numbers!
