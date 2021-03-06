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
- 原始特征几十亿，全连接参数会达到千亿规模
- 每个group内部都有冷门特征，这些特征会被热门特征淹没，导致权重为0




[stackexchange:NN和DNN的区别](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
[Yann LeCun, Bengio, Hinton合著的overview](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)

# 调参经验
## 预处理
特征做zero-centered预处理很重要，高端一点的PCA、whitening没怎么用过。
样本要做随机处理

## 初始化
- 最终收敛效果受初始化影响很大
- 如果将权重全都置为0，那么在反向传播时所有节点梯度都是相同的，执行的更新也是相同的，使得网络不具备不对称性。
- 随机可以产生一些对称性（太小的随机数仍然会导致梯度弥散），但是输出的分布会随输入的个数变化。
- 一般采用layer-wise的均匀分布，即根据每层网络的输入结点个数和输出结点个数，自适应调整随机初始化的取值范围(`range = Sqrt(6/(in_num + out_num))`)
- 《Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》中提到，对ReLU激活函数来说，`w = np.random.randn(n) * sqrt(2.0/n)`是最佳的初始化选择。
- 以上讨论限于Weight，Bias一般初始化为0即可

## 激活函数
无脑ReLu

## 优化函数
Adam训练速度快，但是收敛效果不及SGD+Momentum

## Learning Rate
学习速率过大除了可能会跳过最优解外，在NN中还会造成dead ReLU问题；在与cross-entropy配合使用时会导致loss为Nan。LR的选取与batch size相关，有一种经验之谈是将LR设为1除以batch size，鉴于学习速率对训练的巨大影响，非常值得花时间手动试几组。  
weight dacay应该是标配了吧，每隔一定轮数进行一次衰减。

## 脉冲
随机梯度下降（SGD）是按batch来进行更新，通常来说下降速度比较快，但却容易造成另一个问题，就是更新过程不稳定，容易出现震荡。冲量，就是上一次更新方向所占的权值。引入momentum的idea是很直接的，就是在更新下降方向的时候不仅要考虑到当前的方向，也要考虑到上一次的更新方向，两者加权，某些情况下可以避免震荡。一个小的trick是，当刚开始训练的时候，把冲量设小，或者直接就置为0，然后慢慢增大冲量，有时候效果比较好。

## Batch Size
Batch size会影响训练速度，一般设为100至1000范围内，但并非越大越好。

## Batch Normalization
随着数据流过网络，权重会被逐层缩放，使得输入和输出的分布不一致，这种现象成为**Covariate shift**，且要求learning rate要设置的非常小以防止gradient exploit。Batch Normalization一般放在线性和非线性层之间，归一化 `y= Wx+ b`的结果。加入BN层可以加快收敛速度，提供更小的学习误差；减轻gradient exploit问题，同时可以通过调大learning rate加快学习；

## Regularizations
- L2 L1
- Max Norm：为每个隐层增加一个约束上界c，相当于把w映射到一个半径为c的球面。
- dropout ratio一般设为0.5，对于小数据集上防止过拟合有效。预测时要使用全连接的网络。dropout会在求梯度时引入噪音，因此多个梯度之间可能会相互抵消。为了弥补这种损失，通常需要将learning rate提高（论文中给出的是10到100倍，amazing）或者使用更高的momentum(0.95到0.99)。
。但这样又会导致网络增长过快，因此通常要配合Max Norm使用。

# 参考
- [Batch Normalization](http://shuokay.com/2016/05/28/batch-norm/)
- [深度学习中 Batch Normalization为什么效果好？](https://www.zhihu.com/question/38102762)
-[dropout论文](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Must Know Tips/Tricks in Deep Neural Networks](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)
- [你有哪些deep learning（rnn、cnn）调参的经验？](https://www.zhihu.com/question/41631631)
- [Caffe、TensorFlow、MXnet三个开源库对比](https://chenrudan.github.io/blog/2015/11/18/comparethreeopenlib.html)