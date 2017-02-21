---
title: Udacity DeepLearning Assignment
tags: [online course, machine learning]
---

## 0.安装TensorFlow
1. 安装Homebrew、python、pip、docker、jupyter notebook
1. 尝试了docker和pip两种方式安装TensorFlow。docker library的网站被墙了，可以直接在[某网站](http://7xlgth.com1.z0.glb.clouddn.com/tensorflow.tar)下载了镜像然后load到docker中,也可以按照readme里的流程使用教学docker镜像（ipynb文件同时下好，貌似也会被墙）。最后选择了直接本地安装运行  
2. 安装使用pip安装scipy、sklearn时会特别慢，换成豆瓣的源后速度特别给力。`pip install sklearn -i http://pypi.douban.com/simple --trusted-host pypi.douban.com`酱紫
2. Assignment的要求以ipynb文件的形式放在[GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity)上，可以通过[官方文档](https://www.tensorflow.org/tutorials/)或者[极客学院翻译的TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)了解一下TensorFlow的基本概念及原理：   
	- 使用**tensor**表示数据.在Python中,返回的tensor是`numpy ndarray`对象
	- 使用图(graph)来表示计算任务.图中的节点被称之为**op**(operation). 一个op获得 0个或多个Tensor执行计算, 产生0个或多个Tensor
	- 在被称之为会话(**Session**)的上下文中执行图.会话将图的op分发到诸如CPU或GPU之类的设备上,同时提供执行op的方法并返回产生的Tensor
	- 通过变量(**Variable**)维护状态(计数器等)
	- TensorFlow还提供了**feed**机制, 该机制可以临时替代图中的任意操作中的tensor可以对图中任何操作提交补丁,直接插入一个tensor

## 1.notMNIST
notMNIST是升级版的MNIST，含有A-J10个类别的艺术印刷体字符，难度大于手写数字识别。assignment里提供的地址挂了，可以在[官网](http://yaroslavvb.com/upload/notMNIST/)上直接下载。   
解压后判断每张图片的尺寸是否合规，剔除不合法图片,然后将数据转化成3维数组的形式存储。提供的代码对像素值进行了归一化，但是新下载的数据貌似已经做过归一化了，所以重复归一化会导致所有像素值几乎相同无法进行后续学习（准确率一直是10%，坑了好久），这个要根据解析后的标准差确认一下（之前的标准差只有0.01，对此产生了一点怀疑，然后顺藤摸瓜找到这个问题）。  
使用pyplot将图像展示出来，参见[Pyplot tutorial](http://matplotlib.org/users/pyplot_tutorial.html)   
使用Logistic Regression做简单的训练，观察不同训练样本数量对预测准确度的影响。

```python  
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

