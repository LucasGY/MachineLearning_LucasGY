>微信公众号：**[码蚁LucasGY](#jump_10)**
**关注可了解更多的机器学习基础及工作效率提高技巧。问题或建议，请公众号留言**;
**[如果你觉得对你有帮助，欢迎赞赏](#jump_20)[^1]**

### 内容目录

[TOC]

### 向量的内积的几何意义——投影

两个向量的内积可以被定义为：
$$
\left(a_{1}, a_{2}, \cdots, a_{n}\right) \cdot\left(b_{1}, b_{2}, \cdots, b_{n}\right)^{\top}=a_{1} b_{1}+a_{2} b_{2}+\cdots+a_{n} b_{n}
$$
内积运算将两个向量映射为一个实数，但是这个实数的几何含义却从这个公式中看不出来。
若将内积以另外一种方式写出来，其几何意义会更加明显：
$$
A \cdot B=|A||B| \cos (a)
$$
其中，A、B为n维空间的向量。为方便可视化，将A、B设为二维向量，则
$$
A=\left(x_{1}, y_{1}\right), B=\left(x_{2}, y_{2}\right)
$$
在二维平面上将两个向量可视化：
![2019-10-15-16-09-11](http://pzd8a646b.bkt.clouddn.com/2019-10-15-16-09-11.png)

因此，若B向量的模为1，A与B的内积就是A向B方向上投影的**矢量**长度：
$$
A \cdot B=|A| \cos (a)
$$

### 向量的基底——矩阵相乘的几何含义
我们平时所说的向量一般是从原点出发，如(3,2),但其实他们都是由x轴、y轴长度为1的基底所表示的：
![2019-10-15-19-28-17](http://pzd8a646b.bkt.clouddn.com/2019-10-15-19-28-17.png)
因此，准确来说，向量(x,y)实际上表示为在正交基上的投影：
![2019-10-15-19-30-05](http://pzd8a646b.bkt.clouddn.com/2019-10-15-19-30-05.png)
我们之所以默认选择(1,0)和(0,1)为基，当然是比较方便，因为它们分别是×和y轴正方向上的单位向量，因此就使得二维平面上点 坐标和向量一一对应 ，非常方便。但实际上任何两个线性无关的二维向量都可以成为一组基 ，所谓线性无关在二维平面内可以直 观认为是两个不在一条直线上的向量。
![2019-10-15-19-36-56](http://pzd8a646b.bkt.clouddn.com/2019-10-15-19-36-56.png)
![2019-10-15-19-37-12](http://pzd8a646b.bkt.clouddn.com/2019-10-15-19-37-12.png)
### 方差、协方差与协方差矩阵

#### 方差
$$
S^{2}=\frac{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{2}}{n-1}
$$
Xi为n个样本点，$\bar{X}$为样本均值。直观上看，样本点越分散，方差就应该越大。以两个集合为例，[0, 8, 12, 20]和[8, 9, 11, 12]，两个集合的均值都是10，但显然两个集合的差别是很大的，计算两者的标准差，前者是8.3后者是1.8，显然后者较为集中，故其标准差小一些，标准差描述的就是这种“散布度”。之所以除以n-1而不是n，是因为这样能使我们以较小的样本集更好地逼近总体的标准差，即统计上所谓的“无偏估计”。而方差则仅仅是标准差的平方。

但方差仅仅是测量一个维度上数据的分散程度。协方差就是这样一种用来度量两个随机变量关系的统计量。
#### 协方差
我们可以仿照方差的定义：
$$
\operatorname{var}(\mathrm{X})=\frac{\sum_{\mathrm{i}=1}^{\mathrm{n}}\left(\mathrm{X}_{\mathrm{i}}-\overline{\mathrm{X}}\right)\left(\mathrm{X}_{\mathrm{i}}-\overline{\mathrm{X}}\right)}{\mathrm{n}-1}
$$

来度量各个维度偏离其均值的程度，协方差可以这样来定义：

$$
\operatorname{cov}(\mathrm{X}, \mathrm{Y})=\frac{\sum_{i=1}^{\mathrm{n}}\left(\mathrm{X}_{\mathrm{i}}-\overline{\mathrm{X}}\right)\left(\mathrm{Y}_{\mathrm{i}}-\overline{\mathrm{Y}}\right)}{\mathrm{n}-1}
$$

X,Y分别为两个维度的随机变量，在机器学习中就叫做两个特征。

这里引用"马同学"插图来解释协方差，协方差的公式相当于以$\bar{X}，\bar{Y}$作为原点将二维数据分成四个象限：
![2019-10-15-19-10-26](http://pzd8a646b.bkt.clouddn.com/2019-10-15-19-10-26.png)
协方差相当于是在求每个数据点与均值点形成的坐标轴的面积和：
![2019-10-15-19-13-59](http://pzd8a646b.bkt.clouddn.com/2019-10-15-19-13-59.png)
其中红色的是正的，蓝色的是负的，很明显红色面积多于蓝色面积，因此协方差大于0，推测X,Y成正相关。一般情况下，协方差的绝对值越大，说明两个维度的相关性越强，因此协方差也作为了**相关系数**的分子,如图：
![2019-10-15-19-17-50](http://pzd8a646b.bkt.clouddn.com/2019-10-15-19-17-50.png)

#### 协方差矩阵

协方差也只能处理二维问题，那维数多了自然就需要计算多个协方差，因此有了协方差矩阵，三维的协方差矩阵可以表示为：

$$
C=\left(\begin{array}{ccc}{\operatorname{cov}(x, x)} & {\operatorname{cov}(x, y)} & {\operatorname{cov}(x, z)} \\ {\operatorname{cov}(y, x)} & {\operatorname{cov}(y, y)} & {\operatorname{cov}(y, z)} \\ {\operatorname{cov}(z, x)} & {\operatorname{cov}(z, y)} & {\operatorname{cov}(z, z)}\end{array}\right)
$$

可见，协方差矩阵一般是一个**实对称**的矩阵，而且**对角线是各个维度的方差**。**实对称矩阵在进行变换时有良好的性质（后面会提及）**。

### 由协方差矩阵导出线性PCA降维
上面已经讨论过不同的数据点在不同的的基底上的投影是不一样的，如果基底的数量少于特征的维度，就认为是对原有的数据点进行了降维处理。那么如何去选取新的基底作为数据的投影方向呢？
线性PCA的优化目标：
* **尽可能保留原始的信息量——数据在基底上的投影尽可能分散。**
* **将维后的数据每个维度都不相关。这个很好理解，如果将维后每个维度都是同一个方向，那将维就没有任何意义了。**
因此，我们得到了降维问题的求解方向：

将d维的数据矩阵将为k维，其目标是选择k个单位正交基，使得原始数据变换到这组基上后，各个维度之间的协方差为0，而每个维度的协方差要尽可能地大，越大说明将数据投影到这个基底上面保留的信息量就越多。
#### 推导线性PCA降维
* $X_{m \times d}$为原始数据矩阵，m为样本数，d为特征数；
* C为原始数据的协方差矩阵；
* D为降维后新投影坐标数据的协方差矩阵
* $F_{m \times k}$为变换（降维）后的数据矩阵

已知原始数据的协方差矩阵：
$$
\begin{aligned} C &=\frac{1}{m}(X-\overrightarrow{\mu x})^{\top}_{d \times m}(X-\overrightarrow{\mu x})_{m \times d} \\ &=\frac{1}{m} X^{\prime\top} X^{\prime} \end{aligned}
$$
其中，
$$
X_{m \times d}^{\prime}=\left(X-\mu_{x}\right)=\left[\begin{array}{ccc}{x_{11}-\mu x_{1}} & {\cdots} & {x_{1 d}-\mu x_{d}} \\ {x_{m 1}-\mu x_{1}}&{\cdots} & {x_{m d}-\mu x_{d}}\end{array}\right]_{m \times d}
$$
同理有$F_{m \times k}^{\prime}=\left(F_{m \times k}-\overrightarrow{\mu_{F}}\right)$
设正交变换矩阵为$P_{k \times d}$,行代表单位化的特征向量，行数代表所降的维度，则有
$$
F_{k \times m}^{\top}=P_{k \times d} X_{d x m}^{\top} \Rightarrow F_{k \times m}^{\prime \top}=P_{k \times d} X_{d x m}^{\prime \top} \Rightarrow F^{\prime}=\left(P X^{\prime \top}\right)^{\top}=X^{\prime} P^{\top}
$$
根据线性PCA将为的优化目标，我们只要使得变换后的矩阵D为对角矩阵即可，则有：
$$
\begin{aligned} D &=\frac{1}{m}\left(F-\vec{\mu}_{F}\right)^{T}\left(F-\vec{\mu}_{F}\right) \\ &=\frac{1}{m} F^{\prime \top} F^{\prime}=\frac{1}{m}\left(P X^{\prime \top}\right) X^{\prime} P^{\top} \\ &=P\left(\frac{1}{m} X^{\prime \top} X\right) P^{\top}=P C P^{\top}=\left[\begin{array}{ccc}{\lambda_{1}} & {} \\ {} & {\ddots} \\ {} & {} & {\lambda_{d}}\end{array}\right] \end{aligned}
$$
其中：$\lambda_{1}>\lambda_{2}>...>\lambda_{d}$,为每一个维度的方差。
根据实对称矩阵对角化的过程，我们可知：
P是协方差矩阵的特征向量按照行排列出来的矩阵，每一行都是原始协方差矩阵的特征向量，特征值（方差）越大对应的特征向量（基底）数据在上面的投影由于方差大而越分散，则用P的前k行作为基底，让原始数据在k个正交单位基底上投影，就得到了我们需要的降到k维度的矩阵$F_{k \times m}$。
<a id="jump_10"></a>

### 公众号二维码
![勾越公众号](http://pzd8a646b.bkt.clouddn.com/勾越公众号.jpg)
<a id="jump_20"></a>

### 赞赏码蚁LucasGY

$$
\left(\begin{array}{l}{c t^{\prime}} \\ {x^{\prime}} \\ {y^{\prime}} \\ {z^{\prime}}\end{array}\right)=\left(\begin{array}{cccc}{\gamma} & {-\gamma \beta} & {0} & {0} \\ {-\gamma \beta} & {\gamma} & {0} & {0} \\ {0} & {0} & {1} & {0} \\ {0} & {0} & {0} & {1}\end{array}\right)\left(\begin{array}{l}{c t} \\ {x} \\ {y} \\ {z}\end{array}\right)
$$
