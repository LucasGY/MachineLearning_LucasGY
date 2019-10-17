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

### 上一期自编程实现PCA降维(是马氏距离推导的第一步)
#### 导入原始数据
```python
def loadDataSet(filename, delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)
dataMat = loadDataSet('testSet.txt')
dataMat
plt.scatter(dataMat.A[:,0],dataMat.A[:,1])
```
![2019-10-17-20-43-09](http://pzd8a646b.bkt.clouddn.com/2019-10-17-20-43-09.png)

#### 降成一维数据并投影回原来的二维空间
```python
n_pca = 1 #保留的主成分维度
# 求矩阵每一列的均值
meanVals = np.mean(dataMat, axis=0)
# 数据矩阵每一列特征减去该列特征均值
meanRemoved = dataMat - meanVals
# 计算协方差矩阵，处以n-1是为了得到协方差的无偏估计
# cov(x, 0) = cov(x)除数是n-1(n为样本个数)
# cov(x, 1)除数是n
covMat = np.cov(meanRemoved, rowvar=0)#rowvar=0:行为样本数，列为特征数，计算特征间的协方差
print('协方差矩阵形状：',covMat.shape)

# 计算协方差矩阵的特征值及对应的特征向量
# 均保存在相应的矩阵中
eigVals, eigVects = np.linalg.eig(covMat)
print('特征值：',eigVals)
print('特征向量：',eigVects,'\n一列为一个单位化的特征向量')
# sort():对特征值矩阵排序(由小到大)
# argsort():对特征矩阵进行由小到大排序，返回对应排序后的索引
eigValInd = np.argsort(eigVals)
eigValInd = eigValInd[:: -1] #特征值由大到小的索引
# 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
redEigVects = eigVects[:, eigValInd]
redEigVects
# 将去除均值后的矩阵*压缩矩阵，转换到新的空间，使维度降低为N
lowDDataMat = meanRemoved * redEigVects[:,0:n_pca] #redEigVects为可逆矩阵，redEigVects[:,0:n_pca]不是

# 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
# 此处用转置和逆的结果一样redEigVects.I
reconMat = (lowDDataMat * redEigVects[:,0:n_pca].T) + meanVals 
print(reconMat)
_ = plt.scatter(dataMat.A[:,0],dataMat.A[:,n_pca],label='orign')
_ = plt.scatter(reconMat.A[:,0],reconMat.A[:,n_pca],label='1D')
_ = plt.legend()
```
得到：
![2019-10-17-20-45-19](http://pzd8a646b.bkt.clouddn.com/2019-10-17-20-45-19.png)

**可以看出来，投影回去的点只保留了第一个主成分的信息。**

### 为何需要马氏距离
* 当数据特征之间存在相关性时(X,Y成正相关关系)

A点与B点与数据集中心的欧式距离是相等的，但偏偏A点更像是离群点
![2019-10-17-19-58-15](http://pzd8a646b.bkt.clouddn.com/2019-10-17-19-58-15.png)

* 当数据特征之间相互独立时(X,Y独立同分布)

仍然是同样的结论：
![2019-10-17-20-02-06](http://pzd8a646b.bkt.clouddn.com/2019-10-17-20-02-06.png)

**因此，如果要公平衡量两个点之间的距离，需要将协方差大的那个方向去缩放数据，使得各个维度的方差都为1，再进行欧氏距离的比较，才算是公平的距离计算方式。这就需要上一次讲的PCA降维，来寻找主成分的投影方向。**

### 总结马氏距离的变换过程
由主成分分析可知，由于主成分就是特征向量方向，每个方向的方差就是对应的特征值，所以只需要按照特征向量旋转，然后缩放特征向量倍就可以了，可以得到以下的结果：
所以，计算样本数据的马氏距离分为两个步骤：
	1、坐标旋转
	2、数据压缩
**坐标旋转的目标**：使旋转后的各个维度之间线性无关，所以该旋转过程就是主成分分析的过程。
**数据压缩的目标**：所以将不同的维度上的数据压缩成为方差都是1的的数据集。

### 推导过程
原始数据矩阵$X_{n \times m}$ (n为样本数，m为特征数)：
$$
\begin {matrix} x_{11} & x_{12} &  \cdots  & x_{1m}  \\ x_{21} & x_{22} &  \cdots  & x_{2m}  \\ \vdots  &  \vdots  &  \ddots  &  \vdots   \\ x_{n1} & x_{n2} &  \cdots  & x_{nm}  \\ \end {matrix}
$$
样本的总体均值:
$$
\bf{\mu }_{\bf{X}} = \left( \mu _{X1}, \mu _{X2} \cdots \mu _{Xm} \right)
$$
1. 由上一篇可得知，需要先求原始数据矩阵的协方差矩阵：
$$
\boldsymbol{\Sigma}_{\mathbf{X}}=\mathbf{E}\left\{\left(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}}\right)^{\mathbf{T}}\left(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}}\right)\right\}=\frac{\mathbf{1}}{\mathbf{n}}\left(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}}\right)^{\mathbf{T}}\left(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}}\right)
$$
2. 由协方差矩阵计算其特征值与特征向量：
特征值其实就是每个主成分维度的方差，特征向量其实就是每个主成分维度投影的基底。特征向量组合起来变成变换矩阵$U_{m*m}$,行代表特征向量的个数，也是基底的个数。数据X经过正交变换得到的投影矩阵为：
$$
\boldsymbol{F}^\boldsymbol{T} = (\boldsymbol{F}_\boldsymbol{1}\boldsymbol{,}\boldsymbol{F}_\boldsymbol{2} \cdots \boldsymbol{F}_\boldsymbol{m})^\mathrm{T} = \boldsymbol{U}\boldsymbol{X}^\mathrm{T}
$$
因此，有：
$$
(\bf{F} - \bf{\mu}_{\bf{F}})^{\mathrm{T}} = \bf{U}(\bf{X} - \bf{\mu}_{\bf{X}})^{\mathrm{T}} 
$$
$$
(\bf{F} - \bf{\mu}_{\bf{F}}) = (\bf{X} - \bf{\mu}_{\bf{X}})\bf{U}^{\mathrm{T}}
$$
![2019-10-17-20-34-06](http://pzd8a646b.bkt.clouddn.com/2019-10-17-20-34-06.png)
3. 则一个样本点$\bf{x} = \left( x_1,x_2 \cdots x_m \right)$到均值点$\bf{\mu }_{\bf X} = \left( \mu _{X1},\mu _{X2} \cdots \mu _{Xm} \right)$的马氏距离，等价于求点$\bf{f} = (f_1,f_2 \cdots f_m)$压缩后的坐标值到数据重心压缩后的坐标值$\boldsymbol{\mu }_\boldsymbol{F} = \left( \mu _{F1},\mu _{F2} \cdots \mu _{Fm} \right)$的欧氏距离：
![2019-10-17-20-38-42](http://pzd8a646b.bkt.clouddn.com/2019-10-17-20-38-42.png)

注：可以将上文的$\mu_F$改为任意一个样本点。
### 自编程实现马氏距离求解

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

<a id="jump_10"></a>

### 公众号二维码
![勾越公众号](http://pzd8a646b.bkt.clouddn.com/勾越公众号.jpg)
<a id="jump_20"></a>

### 赞赏码蚁LucasGY
