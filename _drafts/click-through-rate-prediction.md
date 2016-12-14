## Discretization

##Factorization Machine

 

## Model Embedding

采用非线性模型学习intermediate feature，作为ID feature和cross feature的补充，最终输入到线性model来做ctr预估，最早是由facebook提出的，思路大致如下：采用raw features（一般是统计类特征）训练出GBDT模型，获得的所有树的所有叶子节点就是它能够generate出来的特征空间，当每个样本点经过GBDT模型的每一个树时，会落到一个叶子节点，即产生了一个中间特征，所有这些中间特征会配合其他ID类特征以及人肉交叉的特征一起输入到LR模型来做CTR预估。显然，GBDT模型很擅长发掘有区分度的特征，而从根到叶子节点的每一条路径体现了特征组合。对比手工的离散化和特征交叉，模型显然更擅长挖掘出复杂模式，获得更好的效果。

## Wide & Deep Learning

Wide & Deep Learning框架将DNN网络和LR模型并置在同一个网络中，将categorical feature和continuous feature有机的结合在一起，在wide侧通过特征交叉来学习特征间的共现，在deep侧通过将具有泛化能力的categorical feature进行embedding，和continuous feature一起作为DNN网络的输入（可以认为是一种特殊的DNN网络，在网络的最后一层加入了大量的0/1节点），从理论上来说，deep侧可以看做是传统matrix factorization的一种泛化实现，值得注意的是embedding函数是和网络中其他参数通过梯度反向传播共同学习得到。

直观上看，WDL的最大优势是它兼具categorical feature的**记忆能力(memorization)**和continuous feature及DNN隐层带来的**泛化能力(generalization)**。在实践中我们发现，*Deep侧过于稀疏的categorical feature并不会带来效果增益*，我们将出现次数小于某一阈值的特征丢弃，只保留高频id，对于在保留范围以外的新特征，无论打分时还是训练时，embedding值都置位全0；





