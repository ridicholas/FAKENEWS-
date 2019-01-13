#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:55:26 2018

@author: NicholasWolczynski
"""

from sklearn.linear_model import LogisticRegression


log_reg = LogisticRegression(C=1.0, solver='liblinear')
