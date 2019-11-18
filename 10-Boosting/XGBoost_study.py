# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:05:25 2019

@author: 29259
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


from sklearn.datasets.samples_generator import make_classification
# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

sklearn_model_new = xgb.XGBClassifier(max_depth=5,
                                      learning_rate= 0.5, 
                                      verbosity=1, 
                                      objective='binary:logistic',
                                      random_state=1)

sklearn_model_new.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",
                      eval_set=[(X_test, y_test)])