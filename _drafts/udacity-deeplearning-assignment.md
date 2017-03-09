---
title: Udacity DeepLearning Assignment
tags: [online course, machine learning]
---

## 0.安装TensorFlow
1. 安装Homebrew、python、pip、docker、jupyter notebook
1. 尝试了docker和pip两种方式安装TensorFlow。docker library的网站被墙了，可以直接在[某网站](http://7xlgth.com1.z0.glb.clouddn.com/tensorflow.tar)下载了镜像然后load到docker中,也可以按照readme里的流程使用教学docker镜像（ipynb文件同时下好，貌似也会被墙）。最后选择了直接本地安装运行  
2. 使用pip安装scipy、sklearn时会特别慢，换成豆瓣的源后速度特别给力。`pip install sklearn -i http://pypi.douban.com/simple --trusted-host pypi.douban.com`酱紫
2. Assignment的要求以ipynb文件的形式放在[GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity)上，可以通过[官方文档](https://www.tensorflow.org/tutorials/)或者[极客学院翻译的TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)了解一下TensorFlow的基本概念及原理：   
	- 使用**tensor**表示数据.在Python中,返回的tensor是`numpy ndarray`对象
	- 使用图(**graph**)来表示计算任务.图中的节点被称之为**op**(operation). 一个op获得 0个或多个Tensor执行计算, 产生0个或多个Tensor
	- 在被称之为会话(**Session**)的上下文中执行图。会话将图的op分发到诸如CPU或GPU之类的设备上,同时提供执行op的方法并返回产生的Tensor
	- 通过变量(**Variable**)维护状态(计数器等)
	- TensorFlow还提供了**feed**机制, 该机制可以临时替代图中的任意操作中的tensor可以对图中任何操作提交补丁,直接插入一个tensor

## 1.notMNIST
notMNIST是升级版的MNIST，含有A-J10个类别的艺术印刷体字符，难度大于手写数字识别。这个assignment主要用于处理数据。  
- 下载：assignment里提供的地址挂了，可以在[官网](http://yaroslavvb.com/upload/notMNIST/)上直接下载。   
- 格式转换：解压后判断每张图片的尺寸是否合规，剔除不合法图片,然后将数据转化成3维数组的形式存储。  
- 归一化：提供的代码对像素值进行了归一化，但是新下载的数据貌似已经做过归一化了，所以重复归一化会导致所有像素值几乎相同无法进行后续学习（准确率一直是10%，坑了好久），这个要根据解析后的标准差确认一下（之前的标准差只有0.01，对此产生了一点怀疑，然后顺藤摸瓜找到这个问题）。  
- 验证：使用pyplot将图像展示出来，参见[Pyplot tutorial](http://matplotlib.org/users/pyplot_tutorial.html)   
- 存储：将数据集随机划分为训练、测试、验证集，以pickle文件的形式进行存储。pickle模块将对象转化为文件保存在磁盘上，在需要的时候再读取并还原。   
- 训练：使用Logistic Regression做简单的训练，观察不同训练样本数量对预测准确度的影响。

```python  
from sklearn.linear_model import LogisticRegression
import numpy as np

num_samples=[100,300,1000,3000,10000]
lr = LogisticRegression()
trainset=np.reshape(train_dataset,(train_dataset.shape[0],28*28))
testset=np.reshape(test_dataset,(test_dataset.shape[0],28*28))
for k in num_samples:
    lr.fit(trainset[:k],train_labels[:k])
    print (k,lr.score(testset,test_labels))
'''
100 0.6966
300 0.8132
1000 0.8333
3000 0.846
10000 0.8596
'''
```
## 2.fullyConnected
样例中给出了如何使用TensorFlow构建一个多分类的逻辑回归：首先在graph中定义计算过程，然后在session中运行这些op。然后我们参照这个流程构造一层的NN网络，使用ReLu作为激活函数。使用miniBatch随机梯度进行训练，因此训练数据不是一个特定的值，而是占位符**placeholder**，在TensorFlow运行计算时通过feed机制输入这个值。使用batch训练的LR测试集准确率为85.9%，1-layer NN达到91.1%
 
```python
batch_size = 128
layer1_dimension = 1024
num_steps = 3001

graph = tf.Graph()
with graph.as_default():
  # 定义输入输出及参数
  x = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  y = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  valid = tf.constant(valid_dataset)
  test = tf.constant(test_dataset)
  w1 = tf.Variable(tf.truncated_normal([image_size * image_size, layer1_dimension]))
  b1 = tf.Variable(tf.zeros([layer1_dimension]))
  w2 = tf.Variable(tf.truncated_normal([layer1_dimension, num_labels]))  
  b2 = tf.Variable(tf.zeros([num_labels]))
  #定义训练和预测的计算过程 
  l1 = tf.nn.relu(tf.matmul(x, w1) + b1)
  l2 = tf.matmul(l1, w2) + b2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=l2))
  optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

  train_prediction = tf.nn.softmax(l2)
  valid_relu= tf.nn.relu(tf.matmul(valid, w1) + b1)
  valid_prediction = tf.nn.softmax(tf.matmul(valid_relu, w2) + b2)
  test_relu= tf.nn.relu(tf.matmul(test, w1) + b1)
  test_prediction = tf.nn.softmax(tf.matmul(test_relu, w2) + b2)

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  for step in range(num_steps):
  	 # 选取batch数据进行训练
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {x : batch_data, y : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    # 打印训练集、测试集、校验集的loss和准确度
```
## 3. Regularization
- Problem 1：分别对之前训练的LR和NN模型增加了L2正则项
- Problem 2：为了验证正则的作用，在极小的数据集（1024）下反复迭代训练。观察到在没有正则的情况下，训练集的准确率很快收敛到了100%但是测试集准确率只有80.3%；增加正则项后训练集准确率迅速在85%处收敛，测试准确率86.3%。   
- Problem 3：依然使用极小数据集，在隐层后面增加一个dropout层（但是要注意这个dropout层只有训练时用预测时不用），`keep_prob`设为0.5。
- Problem 4：尝试更加复杂的网络（1024\*256\*64）,并使用**learning rate decay**, 效果显著。dropout并没有取得预期的效果。

| L2  |Dropout| 小数据集 | 大数据集 | DNN |
|-----| ----- | ------  | -----  | --- |
| 无 | 无 | 80.3% | 91.1% | 96.5% |
| 有 | 无 | 86.3% | 91% | 96.3% |
| 无 | 有 | 87.1% | 90.8%|  94.4% |
| 有 | 有 | 88%   | 90.8%|  94.4% |


