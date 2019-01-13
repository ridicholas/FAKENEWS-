#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:07:34 2018

@author: NicholasWolczynski
"""


import pandas as pd


train = pd.read_csv("train.csv", encoding='utf_8')  # read in train data
test = pd.read_csv("test.csv", encoding='utf_8')  # read in test data

# Dedup the train and test datasets
train = train.drop_duplicates(subset=['title'], keep='first')
test = test.drop_duplicates(subset=['title'], keep='first')
train = train.drop_duplicates(subset=['text'], keep='first')
test = test.drop_duplicates(subset=['text'], keep='first')

# Convert all text to string
train.text = [str(text) for text in train.text]
test.text = [str(text) for text in test.text]
train.title = [str(title) for title in train.title]
test.title = [str(title) for title in test.title]

# Relabel for clarity (0 = Reliable, 1 = Unreliable)
train['label'] = train['label'].replace(1, 'Unreliable')
train['label'] = train['label'].replace(0, 'Reliable')
test['label'] = test['label'].replace('FAKE', 'Unreliable')
test['label'] = test['label'].replace('REAL', 'Reliable')
