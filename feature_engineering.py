#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:52:06 2018

@author: NicholasWolczynski
"""


import numpy as np
import setup
import re
from textblob import TextBlob
from textstat.textstat import textstat


# %% Working with the title


# Convert title column in dataframe to standalone np array
'''
train_title = np.array(setup.train['title'].astype('U'))
test_title = np.array(setup.test['title'].astype('U'))
'''


# function takes an array and string and
# returns a new array with string_ appended
# to every word in the array
def word_appender(array, to_append):
    new_list = list()
    for row in array:
        new_row = ""
        for word in str.split(row):
            new_row = new_row + word.replace(word, to_append+"_"+word+" ")
        new_list.append(new_row)
    return np.asarray(new_list)


# convert title to new version
"""
train_title = word_appender(train_title, "title")
test_title = word_appender(test_title, "title")
"""


# function will take 2 arrays of strings and combine the strings
# make sure last word of first array ends in a space and arrays are same size


def text_merger(array1, array2):
    new_list = list()
    for i in range(0, len(array2)):
        new_list.append(array1[i]+array2[i])
    return np.asarray(new_list)


# run merger on train and test data
"""
train_text = text_merger(train_title,
                         np.array(setup.train['text'].astype('U')))

test_text = text_merger(test_title, np.array(setup.test['text'].astype('U')))
"""

# %% Urlcount & Article Length


# function that returns # of urls in each instance, and the urls themselves
def url_finder(text_array):
    url_list = list()
    urlcount_list = list()
    for row in text_array:
        urls = re.findall('(?P<url>https?://[^\\s]+)', row)
        urlcount = len(urls)
        url_list.append(urls)
        urlcount_list.append(urlcount)
    return np.asarray(url_list), np.asarray(urlcount_list)


'''
train_urls, train_urlcount = url_finder(train_text)
test_urls, test_urlcount = url_finder(test_text)
'''


# function that will return np array of length of each article
def length_maker(text_array):
    lengths = list()
    for row in text_array:
        lengths.append(len(str(row)))
    return np.asarray(lengths)


"""
train_lengths = length_maker(setup.train['text'])
test_lengths = length_maker(setup.test['text'])
"""


# %% Readability
"""
train_readability = np.asarray(
        [textstat.gunning_fog(text) for text in setup.train.text]
        )

test_readability = np.asarray(
        [textstat.gunning_fog(text) for text in setup.test.text]
        )
"""


# %% Polarity & Subjectivity


# helper methods to calculate polarity and subjectivity
def polarity(text):
    processed = TextBlob(text)
    polarity = processed.sentiment.polarity
    return polarity


def subj(text):
    processed = TextBlob(text)
    subj = processed.sentiment.subjectivity
    return subj


"""
train_polarity = np.asarray(setup.train.text.apply(polarity))
test_polarity = np.asarray(setup.test.text.apply(polarity))

train_subjectivity = np.asarray(setup.train.text.apply(subj))
test_subjectivity = np.asarray(setup.test.text.apply(subj))
"""
