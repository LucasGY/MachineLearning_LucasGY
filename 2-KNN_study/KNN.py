# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 08:08:37 2019

@author: yue.gou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tensorflow as tf 

#%% 模型实现
class KNNClassifier(object):
    """
    KNN Classifier(线性扫描)
    
    Parameters
    -----------
    
    
    Attributes
    ----------
    
    
    """
    def __init__(self,lr=1,epochs=500,optimizer='bgd'):
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        