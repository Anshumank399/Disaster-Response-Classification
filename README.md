# Disaster Response Pipeline Project

### Overview:  
Many messages are recieved during a disaster which are meant for the relief teams. Most of the times the relief teams are divided by the kind of the support that they could provide and the messages needs to be manually segregated. This process could be automated using the machine learning pipeline built in this project. 

### Python Libraries Used:  
> pandas  
> numpy  
> sqlalchemy  
> nltk  
> sklearn  
> joblib  
> flask  
> plotly  

### Scripts:  
1. _model/train_classifier.py_: The ML pipeline for the classification using AdaBoost.  
2. _data/process_data.py_: Clean and process data.
3. _app/run.py_: Flask web app as GUI for model.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.joblib`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

### Sample Output:  
![Sample Output1](https://github.com/Anshumank399/Disaster-Response-Classification/blob/master/Image1.PNG)  
![Sample Output1](https://github.com/Anshumank399/Disaster-Response-Classification/blob/master/Image2.PNG)
