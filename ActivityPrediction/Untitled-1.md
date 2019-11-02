>微信公众号：**[码蚁LucasGY](#jump_10)**
**关注可了解更多的机器学习基础及工作效率提高技巧。问题或建议，请公众号留言**;
**[如果你觉得对你有帮助，欢迎赞赏](#jump_20)[^1]**

这几天看了一下利用传感器数据识别人体动作的数据集，感觉这个对之后做时间序列问题有比较深刻的意义，因此将其实现一下，今后作为参考。

## 公众号二维码
![勾越公众号](http://pzd8a646b.bkt.clouddn.com/勾越公众号.jpg)
<a id="jump_20"></a>

Working-efficiency Github地址：
* [LucasGY efficiency](https://github.com/LucasGY/Working-efficiency)

## 内容目录

[TOC]

### 机器学习的根本问题

机器学习的监督学习的问题中，除了可以分为分类、回归问题，其实也可以被分为时间序列问题与非时间序列问题，如图所示：
![2019-10-25-20-11-02](http://pzd8a646b.bkt.clouddn.com/2019-10-25-20-11-02.png)

时间序列问题其实就是需要根据一段连续时间的数据去预测输出，即many-to-one问题，这个在自然语言处理领域比较常见，然而今天要写的其实是通过一段时间的传感器数据序列，去预测一个人的动作是什么，很显然，我们不可能通过某一时刻的传感器数据而去判断一个人是在坐着还是在上楼。。。

对于这种问题，解决方法有两种，第一种是从时间序列中提取特征，即一个样本为这几个时刻抽离出的特征，如平均值、最大值等等，这方面比较厉害的工具就是**MATLAB和tsfresh**，然后利用传统的机器学习方法或者全连接神经网络便可以解决问题。

第二种方法就是由于深度学习包含了特征工程的部分，因此，利用卷积神经网络或者循环神经网络对时序数据进行特征提取与预测。

![2019-10-25-20-26-53](http://pzd8a646b.bkt.clouddn.com/2019-10-25-20-26-53.png)
### 何时使用 1D-CNN

* 从短（固定长度）片段内提取特征
* 片段内特征位置没有相关性
* 适用数据： 传感器时序数据

### 载入传感器数据并做预处理操作

```python
file_path =r'D:\2-OneDrive\OneDrive\1-Machine Learning\MachineLearning_LucasGY\ActivityPrediction\1-Data\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt'

column_names = ['userid','activity','timestamp','xaxis','yaxis','zaxis'] #原始数据每列含义
raw_data = pd.read_csv(file_path, header=None,names=column_names)
# Last column has a ";" character which must be removed ...
raw_data['zaxis'].replace(regex=True, # to_replace后面是正则表达式
                          inplace=True, #True:直接在原dataframe上改变；false：返回一个series，不改变原dataframe
                          to_replace=r';',value=r'')
raw_data
```
得到的原始数据格式：
![2019-10-25-20-02-28](http://pzd8a646b.bkt.clouddn.com/2019-10-25-20-02-28.png)

userid代表用户；activity代表用户此时的动作；timestamp为时间戳，采样频率为20Hz；后三列分别为x,y,z三个轴向的传感器数据。

由于原始数据含有缺失值，因此要过滤掉：
```python
raw_data.dropna(axis=0, how='any', inplace=True)
raw_data
```
得到：
![2019-10-25-20-33-50](http://pzd8a646b.bkt.clouddn.com/2019-10-25-20-33-50.png)
与上面对比，其实只有一行被过滤掉了。

看一下每个动作包含了多少数据：
```python
fig = plt.figure(figsize=(8,6))
raw_data['activity'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')
```
得到：
![2019-10-25-20-35-23](http://pzd8a646b.bkt.clouddn.com/2019-10-25-20-35-23.png)

可以看出，此数据集类别不平衡，因此后面用准确率这个评估指标并不是很准确。

#### 我们还可以观察各种动作的传感器时序数据
```python
def plot_axis(ax, x, y, title):

    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['xaxis'], 'xaxis')
    plot_axis(ax1, data['timestamp'], data['yaxis'], 'yaxis')
    plot_axis(ax2, data['timestamp'], data['zaxis'], 'zaxis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
for activity in np.unique(raw_data["activity"]):
    subset = raw_data[raw_data["activity"] == activity][:180]
    plot_activity(activity, subset)
```
站立的三轴时序数据：
![2019-10-25-20-39-06](http://pzd8a646b.bkt.clouddn.com/2019-10-25-20-39-06.png)
坐着的三轴时序数据：
![2019-10-25-20-38-48](http://pzd8a646b.bkt.clouddn.com/2019-10-25-20-38-48.png)

我们可以从两张图上看出来坐着与躺着三轴的传感器时间序列数据的表现是不同的，但如果要是具体说出哪里不同，或给定一个时间序列叫人类自己去判断人体的动作，这会显得有点复杂。

### 将字符型离散化标签数字化表示

```python
from sklearn import preprocessing
# Define column name of the label vector
LABEL = "ActivityEncoded"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
raw_data[LABEL] = le.fit_transform(raw_data["activity"].values.ravel())
raw_data
```
得到：
![2019-10-27-16-27-21](http://pzd8a646b.bkt.clouddn.com/2019-10-27-16-27-21.png)

### 划分训练集与测试集

这里只是为了看看效果，并不追求严格的过程方法论原则，将userid大于28的归化为测试集，其他的为验证集与训练集：
```python
# Differentiate between test set and training set
test_data = raw_data[raw_data['userid'] > 28]
train_data = raw_data[raw_data['userid'] <= 28]
```
### z-score标准化训练数据集

```python
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma
# Normalize features for training data set
train_data['xaxis'] = feature_normalize(raw_data['xaxis'])
train_data['yaxis'] = feature_normalize(raw_data['yaxis'])
train_data['zaxis'] = feature_normalize(raw_data['zaxis'])
```
### 将训练数据集预处理成keras需要的格式

**keras需要的格式：**
![2019-10-27-16-38-12](http://pzd8a646b.bkt.clouddn.com/2019-10-27-16-38-12.png)

这里面的处理方法比较简单，选择4秒钟的时序数据作为one sample，由于采样频率为20Hz，因此共有80个时序数据为一个样本，一个样本的形状即为**(1,80,3)**。
代码：
```python
from scipy import stats
def create_segments_and_labels(df, time_steps, step, label_name):

    """
    This function receives a dataframe and returns the reshaped segments
    of x,y,z acceleration as well as the corresponding labels
    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segments
        labels:
    """

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['xaxis'].values[i: i + time_steps]
        ys = df['yaxis'].values[i: i + time_steps]
        zs = df['zaxis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels
x_train, y_train = create_segments_and_labels(train_data,
                                              80,
                                              40,
                                              LABEL)
```
得到的x_train,y_train的形状：
![2019-10-27-16-39-54](http://pzd8a646b.bkt.clouddn.com/2019-10-27-16-39-54.png)

这里最后要做的就是将label向量化：
```python
from tensorflow.keras.utils import to_categorical
num_classes = le.classes_.size #6种姿势
y_train = to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)
y_train[0,:]
```

由于label是有六类动作，因此，向量有六个维度：
```python
from tensorflow.keras.utils import to_categorical
num_classes = le.classes_.size #6种姿势
y_train = to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)
y_train[0,:]
```
### 构建1D卷积神经网路

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
TIME_PERIODS=80
num_sensors = 3
# 1D CNN neural network
model_m = Sequential(name='1DCNN')
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(200, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Flatten())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())
```
得到：
![2019-10-27-16-57-16](http://pzd8a646b.bkt.clouddn.com/2019-10-27-16-57-16.png)
以为卷积与二维卷积的区别就是：一维卷积从源数据的第二轴即时间轴向下滑动，kernel用的数量一般要比二维卷积核要大，记住一个时间段的东西也就越多。如图所示：

![2019-10-27-16-58-38](http://pzd8a646b.bkt.clouddn.com/2019-10-27-16-58-38.png)

由于一维卷积核的长度就是三周传感器数量，为3；宽度为10，filter数量是100；加上偏置，第一层参数的数量为：3\*10\*100+100 = 3100

Maxpooling的窗口大小为3，即三个数据取一个最大值，滑动方式与一维卷积是一样的，都在数据的第二轴即时间轴滑动。

### 训练一维卷积神经网络
```python
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['acc'])

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)
```
这里的callbacks有两个，一个是earlystopping，一个是一个epoch保存一次权值偏置数据。

20%的数据作为验证数据。

由于设置了earlystopping，训练在15Epoch的时候就结束了，得到了损失、准确度曲线：
![2019-10-27-17-47-33](http://pzd8a646b.bkt.clouddn.com/2019-10-27-17-47-33.png)


