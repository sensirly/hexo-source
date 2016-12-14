---
title: 预测模型结果校准
date: 2016-03-16 16:33:43
tags: machine learning
---
# Introduction
由于模型预测时采样不均，或者算法本身的特性（比如SVM和boosting会使结果趋向于呈sigmoid形状的分布；使用独立假设的Naive Bayes会使结果拉向0或1；而NN和bagged tree的bias相对小一些。详见*refer#5*），模型预测值与真实观察值之间往往存在很大的gap。大多数的分类模型，得到的预测结果仅有定序意义，而不能够定量。很多情况下，仅仅得到一个好的AUC值是远远不够的，我们需要得到一个准确的概率值。例如，在优化最大收益的场景下，优化目标是最大化CTR\*CVR\*Price，通过模型分别学到的CTR和CVR的预测值不仅要保序，还要使预测值逼近真是分布才能获得准确的排序结果。
<!-- more -->  
预测校准问题按照预测目标分为4类：  
![](/img/machine_learning/calibration_taxonomy.PNG)   
RP：This kind of regression models are usually referred as "density forecasting" models.例如预测温度在21-24度之间的概率是90%，区间越小表示预测越精准。  

# Solution
对于CD类问题，预测的分类比例与真实数据的比例不一致，通过改变阈值修复全局比例，但是可以会引入更大的误差。这里主要讨论CP类问题的求解。
## 1.Bining  
简单的做法如*refer#1*，将训练集中样本按估计值降序排序，均分成k等分；对于落在某个bin里的新样本，属于某个class的概率等于这个bin中这个class的实例所占的比例。 

*refer#2*中，按照预测值划分等长的n个区间并统计短点处的真实CVR值，对于一个预测值落在[Vi,Vi+1)的新样本，使用两个端点的统计值平滑得出最终预估值。由于某些bin样本过少导致的预测值的非单调性，再使用Isotonic Regression对端点统计值做后处理。

为了更好的拟合预测集的分布，训练集应该被划分成两个，较大的一个用于模型训练，较小的一个预测之后进行分桶矫正。

在样本充足的情况下，bin的个数越多逼近效果越好；因此在不断增大k的过程中，会出现收益的转折点；对于区间划分的方式，可以尝试一下按值和按样本数两种方式相结合，也可以尝试对一些样本充足的bin进行递归分桶，知道达到某一条件停止分裂（比如样本数不足或者不满足单调性时停止）。

## 2.Isotonic Regression 
给定一个无序序列，通过修改每个元素的值得到一个非递减序列 y'，使y和 y' 平方差最小;该算法的假设是映射函数必须是单调的。    
![](/img/machine_learning/isotonic_regression.png)   
Isotonic Regression的一个最为广泛的实现是Pool Adjacent Violators算法，简称PAV算法，主要思想就是通过不断合并、调整违反单调性的局部区间，使得最终得到的区间满足单调性。 
![](/img/machine_learning/PAV.PNG)   
Isotonic Regression通常作为辅助其他方法修复因为数据稀疏性导致的矫正结果不平滑问题 

## 3.参数拟合分布：
以预估值作为变量，观测值作为目标，用回归算法拟合参数。 
- Platt’s Method(*refer#1* *refer#5*)：用sigmoid函数将原始输出值映射成概率值p=1/(1+e^(A*p+B)),参数A、B通过大似然法获取。适用于SVM、boosting等算法的结果矫正。  
- Google的CTR预测(*refer#3*)中尝试了Poisson  regression做预估CTR和真实CTR的映射。更为精准的校准：分段学习参数，然后使用Isotonic Regression平滑结果。

## 4.其他
- *refer#6*中提出了针对决策树矫正的方案：Curtailment解决叶子节点样本不足的置信问题 + Laplace平滑解决C4.5等算法产生同质化节点导致预估值趋向两端的情况。


# Reference
1. [Calibration of Machine Learning Models](http://users.dsic.upv.es/~flip/papers/BFHRHandbook2010.pdf)
2. [Estimating Conversion Rate in Display Advertising from Past Performance Data](https://pdfs.semanticscholar.org/379a/1c6d825f957f030cda8babc519738c224ca3.pdf)
3. [Ad Click Prediction : a View from the Trenches Categories and Subject Descriptors](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)
4. [Efficient regularized isotonic regression with application to gene-gene interaction search](http://arxiv.org/pdf/1102.5496.pdf)
5. [Predicting good probabilities with supervised learning](http://www.datascienceassn.org/sites/default/files/Predicting%20good%20probabilities%20with%20supervised%20learning.pdf)
6. [Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers](http://cseweb.ucsd.edu/~elkan/calibrated.pdf)
7. [Transforming classifier scores into accurate multiclass probability estimates](http://120.52.72.36/www.research.ibm.com/c3pr90ntcsf0/people/z/zadrozny/kdd2002-Transf.pdf)
8. [On Calibrated Predictions for Auction Selection Mechanisms](http://arxiv.org/pdf/1211.3955.pdf)
9. [Practical Lessons from Predicting Clicks on Ads at Facebook](http://www.herbrich.me/papers/adclicksfacebook.pdf)


