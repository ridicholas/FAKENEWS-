#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:12:33 2019

@author: NicholasWolczynski
"""


import pandas as pd
import numpy as np


def evaluator(fitting, testing, train_y, test_y, model, strength):
    predictions = list()
    prediction_probs = list()
    precision = list()
    pred = list()
    true = list()
    for i in range(0, len(testing)):
        true_pos = 0
        pred_pos = 0
        model.fit(fitting[i], train_y)
        predictions.append(model.predict(testing[i]))
        prediction_probs.append(model.predict_proba(testing[i]))
        for j in range(0, len(prediction_probs[-1])):
            if prediction_probs[-1][j][1] >= strength:
                pred_pos += 1
                if test_y[j] == 1:
                    true_pos += 1
        precision.append(true_pos/pred_pos)
        pred.append(pred_pos)
        true.append(true_pos)
    new_precision = pd.DataFrame({'pred_pos': pred,
                                  'true_pos': true,
                                  'precision': precision})
    return np.asarray(predictions), np.asarray(prediction_probs), new_precision
