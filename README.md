<img src="Udacity_logo.png" align="right" />

# Disaster Response Pipeline Project
## by Iman Babaei
date: 06.10.2022


### Instructions:

This project tries to come up with a model which categorizes messages with the help of NLP and machine learning techniques. The input data are messages written by people on social media during a crisis like flood or hurricane. This data is provided by [Figur8](https://figur8tech.com/), partner of [Udacity](https://www.udacity.com/) for this project. They have labeled the raw data and allocated them to the corresponding categories.  
This application helps to categorize what people need and mean when posting a message online and can help the first aid providers to better use their time and supplies. Especially, at the beginning of the crisis when they are super busy and time is sparse. With some further modification, it can be used to provide automatic replies as well and provide some information like telephone numbers of the authorities to contact or links to safe places and already provided helps.

Below is an explanation of the files in the repository:

- app
    - template
        - master.html # main page of web app
        - go.html # classification result page of web app
    - run.py # Flask file that runs app
- data
    - .ipynb_checkpoints 
        - ETL Pipeline Preparation-checkpoint.ipynb # checkpoints of the jupyter notebook
    - DisasterResponse.db.db # database to save clean data to
    - ETL Pipeline Preparation.ipynb # jupyter notebook used for ETL
    - disaster_categories.csv # data to process
    - disaster_messages.csv # data to process
    - process_data.py # python ETL app
- models
    - .ipynb_checkpoints
        - ML Pipeline Preparation-checkpoint.ipynb # checkpoints of the jupyter notebook
    - ML Pipeline Preparation.ipynb # jupyter notebook used for ML Pipeline
    - model.pkl # saved model
    - train_classifier.py
- .gitignore # template file for git to ignore some changes
- README.md
- Results.PNG # results snapshot
- Udacity_logo.png # Udacity logo
- bar_message_categories.png # distribution of all messages in the data
- bar_top10_categories.png # distribution of top 10 categories 
- pie_message_genres.png # pie chart of the genres

To run the apps follow the below steps:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Clarification 

- First I used the ["ETL Pipeline Preparation.ipynb"](https://github.com/ImiGit/disaster-response-pipeline-project/blob/main/data/ETL%20Pipeline%20Preparation.ipynb) Jupyter notebook template provided by Udacity to make the ETL pipeline, step by step. I have made comments and clarification in the notebooks.

- Then, the [process_data.py](https://github.com/ImiGit/disaster-response-pipeline-project/blob/main/data/process_data.py) is completed, so it can be run as:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

- Third step was to write ML pipeline. Similar to ETL one, [ML Pipeline Preparation.ipynb](https://github.com/ImiGit/disaster-response-pipeline-project/blob/main/models/ML%20Pipeline%20Preparation.ipynb) Jupyter notebook provided by Udacity was used. Different classification algorithms were tested and evaluated and SGDClassifier was chosen at the end, based on the classification metrics and speed.

| Precision | recall | f1-score |
| --- | --- | --- |
| 0.94 | 0.95 | 0.94| 

The best model is saved as a pickle file. For further notes about the algorithms and parameters, you can refer to the [ML Pipeline Preparation notebook](https://github.com/ImiGit/disaster-response-pipeline-project/blob/main/models/ML%20Pipeline%20Preparation.ipynb)

- [train_classifier.py](https://github.com/ImiGit/disaster-response-pipeline-project/blob/main/models/train_classifier.py) is completed using the results of the ML Pipeline notebook.

- [run.py app](https://github.com/ImiGit/disaster-response-pipeline-project/blob/main/app/run.py) is modified, so it can access the database and model and run some commands to illustrate the data using flask and plotly. Below you can see three snapshots of these modifications.

<img src="pie_message_genres.png" align="Center" />
<img src="bar_top10_categories.png" align="Center" />
<img src="bar_message_categories.png" align="Center" />

- Last, I changed the [go.html file](https://github.com/ImiGit/disaster-response-pipeline-project/blob/main/app/templates/go.html) in the templates a bit, so that it shows all the category results of the model first, and then below that shows all the possible categories and positive results of the model.

<img src="Results.PNG" align="Center" />