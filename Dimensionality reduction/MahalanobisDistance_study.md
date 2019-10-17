>微信公众号：**[码蚁LucasGY](#jump_10)**
**关注可了解更多的机器学习基础及工作效率提高技巧。问题或建议，请公众号留言**;
**[如果你觉得对你有帮助，欢迎赞赏](#jump_20)[^1]**

谢谢给位小伙伴厚爱，有人提出是否在写的过程中show一下代码，因此从本文章开始，用到的代码都会show出来，代码和笔记同时放在我的Github上，欢迎大家继续关注我的公众号。这里的关注指的是点击关注公众号哈！

Github地址：
* [LucasGY ML](https://github.com/LucasGY/MachineLearning_LucasGY)

上一篇线性PCA：
* [LucasGY 线性PCA](https://mp.weixin.qq.com/s/SrVBD2URRg7uLnd42kysXQ)

### 内容目录

[TOC]

### 为何需要马氏距离

### 为何需要马氏距离

### 一步一步实现马氏距离求解

#### 载入原始数据
X,Y轴分别为两个维度的特征，可以看出来两个特征成正相关关系，在平面内取与这些数据均值欧式距离相等的两个点，并计算两个点到均值点的实际欧式距离。
```python
def loadDataSet(filename, delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)
dataMat = loadDataSet('testSet.txt')
dataMat
datamean = np.array([dataMat.A[:,0].mean(),dataMat.A[:,1].mean()])
data1 = np.array([dataMat.A[:,0].mean()-3,dataMat.A[:,1].mean()+3])
data2 = np.array([dataMat.A[:,0].mean()+3,dataMat.A[:,1].mean()+3])

fig = plt.figure(dpi=150)
plt.scatter(dataMat.A[:,0],dataMat.A[:,1])
plt.scatter(datamean[0],datamean[1],s=30,c='r',label='mean')
plt.scatter(data1[0],data1[1],s=30,c='r',marker='x',label='data1')
plt.scatter(data2[0],data2[1],s=30,c='r',marker='^',label='data2')
plt.grid(linestyle ='--')
plt.legend()
plt.title('与均值点的欧式距离：\n'+'data1:{}，data2:{}'.format(np.sqrt(np.sum(np.square(data1-datamean))),np.sqrt(np.sum(np.square(data2-datamean)))),fontproperties='SimHei',fontsize=13)
```
![2019-10-17-17-37-01](http://pzd8a646b.bkt.clouddn.com/2019-10-17-17-37-01.png)

#### 将数据集投影到各个主成分上

```python
# 求矩阵每一列的均值
meanVals = np.mean(dataMat, axis=0)
# 数据矩阵每一列特征减去该列特征均值
meanRemoved = dataMat - meanVals
# 计算协方差矩阵，处以n-1是为了得到协方差的无偏估计
covMat = np.cov(meanRemoved, rowvar=0)#rowvar=0:行为样本数，列为特征数，计算特征间的协方差

# 计算协方差矩阵的特征值及对应的特征向量
eigVals, eigVects = np.linalg.eig(covMat)
# argsort():对特征矩阵进行由小到大排序，返回对应排序后的索引
eigValInd = np.argsort(eigVals)
eigValInd = eigValInd[:: -1] #特征值由大到小的索引
# 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
redEigVects = eigVects[:, eigValInd]
# 将去除均值后的矩阵*压缩矩阵，转换到新的空间，使维度降低为N
lowDDataMat = meanRemoved * redEigVects[:,:] #redEigVects为可逆矩阵，redEigVects[:,0:n_pca]不是

datamean_new = (datamean[None,:]-meanVals)* redEigVects[:,:]
data1_new = (data1[None,:]-meanVals)* redEigVects[:,:]
data2_new = (data2[None,:]-meanVals)* redEigVects[:,:]

fig = plt.figure(dpi=150)
plt.scatter(lowDDataMat.A[:,0],lowDDataMat.A[:,1])
plt.scatter(datamean_new[0,0],datamean_new[0,1],s=30,c='r',label='mean')
plt.scatter(data1_new[0,0],data1_new[0,1],s=30,c='r',marker='x',label='data1')
plt.scatter(data2_new[0,0],data2_new[0,1],s=30,c='r',marker='^',label='data2')
plt.grid(linestyle ='--')
plt.legend()
plt.axis('equal')
plt.title('与均值点的欧式距离：\n'+'data1:{}，data2:{}'.format(np.sqrt(np.sum(np.square(data1_new-datamean_new))),np.sqrt(np.sum(np.square(data2_new-datamean_new)))),fontproperties='SimHei',fontsize=13)
```
![2019-10-17-17-40-18](http://pzd8a646b.bkt.clouddn.com/2019-10-17-17-40-18.png)

**注意：此时距离是不变的，因为各个主成分是正交的关系，所以投影后相当于只是旋转了坐标轴**

#### 将每个主成分的维度除以标准差
前面求得的特征向量其实就是每个主成分轴对应的方差，PCA那一节讲过，所以这里将数据沿着主成分的维度除以自身的标准差，以得到标准化距离。

```python
reg_eigVals = eigVals[eigValInd] #特征值由大到小排序
lowDDataMat_sca = lowDDataMat/np.sqrt(reg_eigVals)
datamean_new_sca = datamean_new/np.sqrt(reg_eigVals)
data1_new_sca = data1_new/np.sqrt(reg_eigVals)
data2_new_sca = data2_new/np.sqrt(reg_eigVals)

fig = plt.figure(dpi=150)
plt.scatter(lowDDataMat_sca.A[:,0],lowDDataMat_sca.A[:,1])
plt.scatter(datamean_new_sca[0,0],datamean_new_sca[0,1],s=30,c='r',label='mean')
plt.scatter(data1_new_sca[0,0],data1_new_sca[0,1],s=30,c='r',marker='x',label='data1')
plt.scatter(data2_new_sca[0,0],data2_new_sca[0,1],s=30,c='r',marker='^',label='data2')
plt.grid(linestyle ='--')
plt.legend()
plt.axis('equal')
plt.title('与均值点的欧式距离：\n'+'data1:{}，data2:{}'.format(np.sqrt(np.sum(np.square(data1_new_sca-datamean_new_sca))),np.sqrt(np.sum(np.square(data2_new_sca-datamean_new_sca)))),fontproperties='SimHei',fontsize=13)
```
![2019-10-17-17-44-37](http://pzd8a646b.bkt.clouddn.com/2019-10-17-17-44-37.png)
#### 验证马氏距离计算的准确性

```python
from scipy.spatial import distance

distance.mahalanobis(data1[None,:],datamean[None,:] , np.mat(covMat).I)
distance.mahalanobis(data2[None,:],datamean[None,:] , np.mat(covMat).I)
```
![2019-10-17-17-35-01](http://pzd8a646b.bkt.clouddn.com/2019-10-17-17-35-01.png)

