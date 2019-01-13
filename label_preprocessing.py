#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:26:26 2018

@author: NicholasWolczynski
"""


import setup
from sklearn import preprocessing


# Create lable encoder
class_y = ['Unreliable', 'Reliable']
le = preprocessing.LabelEncoder()
le.fit(class_y)


# Encode labels for both train and testing data
train_y = le.transform(setup.train['label'])
test_y = le.transform(setup.test['label'])


# Check to see if it worked
print(le.transform(['Unreliable', 'Reliable', 'Reliable']))
print(train_y)
print(test_y)
