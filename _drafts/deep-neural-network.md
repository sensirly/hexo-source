## 为什么多层网络是有效的?
如果神经元是非线性的，比如是平方函数，那么两层就能表示四次方，三层就能表示六次方，更深的网络可以表达更复杂的函数。

## 为什么不是层数越多越好？
因为非常难以训练。神经网络常用的训练方法是梯度下降，当层数过多时会陷入**vanishing gradients**问题：反向传播通过链式法则求各层网络的导数值，深层网络使得链路过长导致斜率过小难以更新。这是2000年之前神经网络发展缓慢的重要原因。

## 什么促使神经网络死而复生
主要源于最近几年训练方法上的研究成果，以及计算能力（GPU、并行等）的提高和训练数据的增加   
- Hinton老爷子在2006年发表了几篇重大突破的论文，通过在正式训练前进行非监督的pre-training来进行高效地训练，Hinton将之称为**deep belief networks**。   
- 2010年Martens发表了一种称为**Hessian-free**的方法（通过二阶导数求解梯度）击败了pre-training的神经网络。      
- 2013年Sutskever et al.提出了一些优雅的改进和trick，使得随机梯度下降法击败Hessian-free的方法。   

## 为什么输入层要先做embedding
- 原始特征几十亿，之前全连接参数会达到千亿规模
- 每个group内部都有冷门特征，这些特征会被热门特征淹没，导致权重为0




[stackexchange:NN和DNN的区别](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
[Yann LeCun, Bengio, Hinton合著的overview](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)

http://www.52cs.org/?p=1046
《Deep Learning over Multi-field Categorical Data – A Case Study on User Response Prediction in Display Ads》

FM的前半部分可以看成是Logistic Regression，后半部分是Feature Interaction。



## 脉冲
随机梯度下降（SGD）是按batch来进行更新，通常来说下降速度比较快，但却容易造成另一个问题，就是更新过程不稳定，容易出现震荡。
引入momentum的idea是很直接的，就是在更新下降方向的时候不仅要考虑到当前的方向，也要考虑到上一次的更新方向，两者加权，某些情况下可以避免震荡。冲量，就是上一次更新方向所占的权值。
一个小的trick是，当刚开始训练的时候，把冲量设小，或者直接就置为0，然后慢慢增大冲量，有时候效果比较好。

## Weight Decay
## Learning Rate
## Batch Size

## Batch Normalization
http://shuokay.com/2016/05/28/batch-norm/