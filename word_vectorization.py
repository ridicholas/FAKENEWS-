#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:14:06 2018

@author: NicholasWolczynski
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import pandas as pd


# %%
# initiate parameter values through which we will iterate
df_iter = [[.05, .95], [.1, .9], [.2, .8]]


# helper function that takes text, vectorizer, and existing feature arrays and
# transforms text depending on vectorizer, then returns final feature list and
# number of features. To be used only on creating training representation
def train_feature_maker(text,
                        vectorizers,
                        url,
                        length,
                        text_readability,
                        text_polarity,
                        text_subjectivity):
    features = list()
    sizes = list()
    for vectorizer in vectorizers:
        features.append(hstack((vectorizer.fit_transform(text),
                                url[:, None],
                                length[:, None],
                                text_polarity[:, None],
                                text_subjectivity[:, None],
                                text_readability[:, None])).toarray())
        sizes.append(features[-1].shape)
    return features, sizes


# Similar to train_feature_maker, but instead of using fit_transform on text
# with vectorizer, uses transform, which uses the existing fit from train data
def test_feature_maker(text,
                       vectorizers,
                       url,
                       length,
                       text_readability,
                       text_polarity,
                       text_subjectivity):
    features = list()
    sizes = list()
    for vectorizer in vectorizers:
        features.append(hstack((vectorizer.transform(text),
                                url[:, None],
                                length[:, None],
                                text_polarity[:, None],
                                text_subjectivity[:, None],
                                text_readability[:, None])).toarray())
        sizes.append(features[-1].shape)
    return features, sizes


# function that first creates a list of vectorizers, then creates features from
# the text for each vectorizer and combines those features with existing
# features. Final result is a dataframe of different feature representations
def vectorizer_maker(text,
                     df_iter,
                     n_gram_max,
                     url, length,
                     text_readability,
                     text_polarity,
                     text_subjectivity):
    vectorizers = list()
    dfs = list()
    gram = list()
    Type = list()
    for t in ['count', 'tfidf']:
        for low in range(1, n_gram_max+1):
            for high in range(low, n_gram_max+1):
                for df in df_iter:
                    dfs.append(df)
                    gram.append(str(low) + "-" + str(high))
                    Type.append(t)
                    if t == 'count':
                        vectorizers.append(
                                CountVectorizer(min_df=df[0],
                                                max_df=df[1],
                                                max_features=5000,
                                                stop_words='english',
                                                ngram_range=(low, high)
                                                )
                                )
                    else:
                        vectorizers.append(
                                TfidfVectorizer(min_df=df[0],
                                                max_df=df[1],
                                                max_features=5000,
                                                stop_words='english',
                                                ngram_range=(low, high)
                                                )
                                          )
    features, shape = train_feature_maker(text,
                                          vectorizers,
                                          url,
                                          length,
                                          text_readability,
                                          text_polarity,
                                          text_subjectivity)
    frame = pd.DataFrame({'grams': gram,
                          'dfs': dfs,
                          'vectorizers': vectorizers,
                          'features': features,
                          'shape': shape,
                          'type': Type
                          })
    return frame


# %% Create feature set using vectorizer and feature creation functions
"""
feature_set = vectorizer_maker(train_text,
                               df_iter, 2,
                               train_urlcount,
                               train_lengths,
                               train_readability,
                               train_polarity,
                               train_subjectivity)
"""
