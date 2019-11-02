# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:35:55 2019

@author: yue.gou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tensorflow as tf 
#import seaborn as sns
# %%自定义计算层与计算模型
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.units = units
        
    def build(self,input_shape):
        self.W = tf.Variable(initial_value=[[0.6]],shape=[input_shape[-1],self.units])
        self.b = tf.Variable(initial_value=3.,)
#        self.W = self.add_weight(name='W',
#                                   shape=[input_shape[-1],self.units],
#                                   initializer=tf.zeros_initializer())
#        self.b = self.add_weight(name='b',
#                                   shape=[self.units],
#                                   initializer=tf.zeros_initializer())
        
    def call(self,inputs):
        y_pred = tf.matmul(inputs,self.W) + self.b
        return y_pred
    
class LinearRegressionModel(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        self.layer = LinearLayer(units=1)
        
        
    def call(self,inputs):
        output = self.layer(inputs)
        return output

# %%实例化模型
model = LinearRegressionModel()
# %%加载源数据
data = pd.read_csv(r'D:\2-OneDrive\OneDrive\1-Machine Learning\MachineLearning_LucasGY\LinearRegression\Advertising.csv',index_col=0) # csv文件第一列作为dataframe的index

X_train = data.TV.values[:,None] #200行1列
y_train = data.sales.values[:,None] #200行1列
# %%执行梯度下降，并可视化
#动态图可视化
def f(W,B):
    out_array = []
    z = [i for i in zip(W.flat, B.flat)]
    for buf in z:
        y_pred = X_train*buf[0] + buf[1]
        Loss = np.mean(np.square(y_pred - y_train))
        out_array.append(Loss)
        """
    for w_index,w_ in enumerate(w):
        for b_index,b_ in enumerate(b):
            y_pred = X_train*w_ + b_
            Loss = np.mean(np.square(y_pred - y_train))
            out_array[w_index][b_index] = Loss
            """
    return out_array
w = np.linspace(-0.1,0.6,300)
b = np.linspace(-0.1,6,300)
W, B = np.meshgrid(w, b)
Loss_array = np.array(f(W,B)).reshape(len(W),len(B))

fig = plt.figure(figsize=(15,10))
ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2,projection='3d')
ax0.scatter(X_train.flatten(), y_train.flatten()) #真实数据，不能删除
ax1.plot_surface(W, B, Loss_array, cmap='viridis')
ax1.set_xlabel('W')
ax1.set_ylabel('B')
ax1.set_zlabel('Loss');
plt.ion()
plt.show()

num_epoch = 5000
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
for i in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(X_train)
        Loss = tf.reduce_mean(tf.square(y_pred - y_train))
    grads = tape.gradient(Loss, model.variables)
    _ = optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    if i % 10 == 0:
        # 先清除，后画出pre（因此需要排除第一次错误，lines第一次没生成，用try/except）
        try:
            ax0.lines.remove(lines[0])######important!!!
        except Exception:
            pass
        # plot the prediction
        lines = ax0.plot(X_train.flatten(), y_pred.numpy().flatten(), 'r-', lw=5)#lw:线宽
#        print(Loss)
        ax1.scatter(model.variables[0].numpy().flatten(),
                    model.variables[1].numpy().flatten(),
                    Loss.numpy(),s=50,c='red')
        plt.pause(0.1)
        # 关闭交互模式
        plt.ioff()
        # 图形显示
        plt.show()
print(model.variables)

"""
def f(w, b):
    out_array = np.zeros((len(w),len(b)))
    for x_index,x in enumerate(w):
        for y_index,y in enumerate(b):
            out_array[x_index][y_index] = tf.reduce_mean(tf.square(x*X_train+y - y_train)).numpy()
    
    return out_array
w = np.linspace(-model.variables[0].numpy().flatten()-5, model.variables[0].numpy().flatten()+5, 300)
b = np.linspace(-model.variables[1].numpy().flatten()-5, model.variables[1].numpy().flatten()+5, 300)
W, B = np.meshgrid(w, b)
Z = f(w, b)






for i in range(10000):
    with tf.GradientTape() as tape:
        y_pred = model(X_train)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y_train))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    
print(model.variables)
"""