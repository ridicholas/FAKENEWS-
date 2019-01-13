# FAKE NEWS!

This repository contains my work on identifying fake news. The best place for commentary on my thought process, methodology, and results is on the uploaded Jupyter notebook. However, I have made the code and datasets I used available in case you'd like to run it yourself.

## Data 

Training data: https://www.kaggle.com/c/fake-news/data

Test data: https://github.com/GeorgeMcIntire/fake_real_news_dataset

## Code! 

#### setup.py
Contains code to read in the data (make sure dataset is in your wd), and does first-stage preprocessing (dropping dupes, converting to string, creating clear and uniform labels). 

#### label_preprocessing.py
Creates training and testing label arrays that will be used later on.

#### feature_engineering.py
Functions to create the following features: url_count, article length, article polarity, subjectivity, and readability. Polarity, subjectivity, and readability taken from the textblob and textstat packages: https://github.com/shivam5992/textstat

#### word_vectorization.py
A continuation of feature engineering, but dealing with creating word vectors. Functions that iteratively create various vectorizers (count vs tfidf, experimenting with min/max df ranges, and trying different uni-bi gram combinations). These vectorizers are then used to create different feature vectors, and combine all of the attributes, vectorizers, and final features into a single dataframe. 

#### modeling.py
Not much here since all I used was a simple logistic regression model.

#### evaluation.py
Contains the evaluator function. This will take our sets of training feature representations + their corresponding test feature representations (a test feature representation has to be made of the features from the training set), the training and test targets, the model of choice (in this case log_reg), and a desired probability threshold for making predictions. This function iterates through each feature representation, fitting the model onto the training data and using it to make predictions using the testing input. The model then returns the predictions (using 50% probability threshold), the prediction probability, and a data frame containing final results for each feature representation using the input probability threshold. Final results = (# of predicted positive, # of correctly predicted positive, precision). 

***Some of the .py files contain commented out sections of code which are used in the jupyter notebook. 





