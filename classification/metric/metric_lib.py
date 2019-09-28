#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sklearn.metrics as metrics

class Metric(object):
    def __init__(self, params={}):
        self.latest_score = None
        self.params = params 

    def calc_score():
        raise NotImplementedError

    def add_tensorboard(self, writer, name, epoch):
        if self.latest_score is not None:
            writer.add_scaler(name, self.latest_score, epoch)
        else:
            print('[Warning] score is not calcurated yet')

class AccuracyScore(Metric):

    def calc_score(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
   
        score = metrics.accuracy_score(y_true, y_pred, **self.params)
        self.latest_score = score

        return score
