# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:56:11 2019

@author: yue.gou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tensorflow as tf 

np.random.seed(1)

#%% 简单实现
class PerceptronModel(object):
    """
    Perceptron Classifier
    
    Parameters
    -----------
    lr: float
        Learning rate (between 0.0 to 1.0)
        
    epochs: int
        Passes(epochs) over the training set.
        
    optmizer: str
        'bgd':批量梯度下降
        'sgd':随机梯度下降
        
    Attributes
    ----------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Number of misclassification in every epoch.
        
    """
    def __init__(self,lr=1,epochs=500,optimizer='sgd2'):
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
    def predict(self,X):
        """
        Predict the output with X.
        
        """
        y_pre_buf = np.dot(X,self.w_[1:])+self.w_[0]
        y_pre = np.where(y_pre_buf>0., 1, -1)
        return y_pre
    def fit(self,X,y):
        """
        Fit method for training data.
        
        Para
        ----------
        X: {array-like},shape=[n_samples,n_features]
        
        y:{array-like},shape=[n_features]
        
        Returns
        ----------
        self:object
        
        """
        # 可视化：静态部分，只显示原数据点
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X[:,0].flatten(), X[:,1]) #真实数据，不能删除
        plt.ion()
        plt.show()
        
        self.w_ = np.zeros(1+X.shape[1])
        self.errors = []
        
        if self.optimizer == 'sgd1':
            for _ in range(self.epochs):
                index = np.arange(X.shape[0])
                shuffle_index = np.random.permutation(index)
                for xi,target in zip(X[shuffle_index,:],y[shuffle_index]):
                    update = self.lr * 0.5*(target-self.predict(xi)) #预测正确为0，预测有误就是-2/+2，为了与书上求导公式一致
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    if update != 0 :
                        break
            return self
            pass
        elif self.optimizer == 'sgd2':
            for _ in range(self.epochs):
                errors = 0
                for xi,target in zip(X,y):
                    update = self.lr * 0.5*(target-self.predict(xi)) #预测正确为0，预测有误就是-2/+2，为了与书上求导公式一致
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0)
                    if errors != 0:
                        
                        print(self.w_[1:], self.w_[0])
                        #更新图片
                        try:
                            ax.lines.remove(lines[0])######important!!!
                        except Exception:
                            pass
                        # plot the prediction
                        x_ = np.arange(0,10)
                        lines = ax.plot(x_, -(self.w_[1]*x_+self.w_[0])/self.w_[2], 'r-', lw=2)#lw:线宽
    
                        plt.pause(1)
                        # 关闭交互模式
                        plt.ioff()
                        # 图形显示
                        plt.show()
                self.errors.append(errors)
                if errors == 0:
                    break
            return self
        elif self.optimizer == 'bgd':
            for _ in range(self.epochs):
#                y_pre = self.predict(X)
#                X_err = X[y_pre!=y]
                update = self.lr * 0.5*(y-self.predict(X)) #预测正确为0，预测有误就是-2/+2，为了与书上求导公式一致
                self.w_[1:] += np.sum(update[:,None] * X)
                self.w_[0] += np.sum(update)
                #更新图片
                try:
                    ax.lines.remove(lines[0])######important!!!
                except Exception:
                    pass
                # plot the prediction
                x_ = np.arange(0,10)
                lines = ax.plot(x_, -(self.w_[1]*x_+self.w_[0])/self.w_[2], 'r-', lw=2)#lw:线宽

                plt.pause(1)
                # 关闭交互模式
                plt.ioff()
                # 图形显示
                plt.show()
                if (update==0).all():
                    return self
        else:
            pass
        
Xtrain = np.array([[3,3],[4,3],[1,1]])  
y_train = np.array([1,1,-1])
model = PerceptronModel()


model.fit(Xtrain,y_train)
"""
# %%自定义计算层与计算模型
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.units = units
        
    def build(self,input_shape):
#        self.W = tf.Variable(initial_value=[[0.6]],shape=[input_shape[-1],self.units])
#        self.b = tf.Variable(initial_value=3.,)
        self.W = self.add_weight(name='W',
                                   shape=[input_shape[-1],self.units],
                                   initializer=tf.zeros_initializer())
        self.b = self.add_weight(name='b',
                                   shape=[self.units],
                                   initializer=tf.zeros_initializer())
    def activator(self,input_value):
        if input_value>=0:
            return 1
        else:
            return -1
    def call(self,inputs):
        y_pred = self.activator(tf.matmul(inputs,self.W) + self.b)
        return y_pred
class PerceptronModel(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        self.layer = LinearLayer(units=1)
        
        
    def call(self,inputs):
        output = self.layer(inputs)
        return output
    
    def fit(self,):
        
        

# %%实例化模型
model = PerceptronModel()

#%% loss func
class PerceptronModelLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        
        return tf.reduce_mean(tf.square(y_pred - y_true))
"""