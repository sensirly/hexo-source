---
title: Classification Model Evaluation 
tags:
  - machine learning
date: 2016-09-11 10:29:05
---
AUC和F1常被用来评测二分类算法的优劣。
# Precision, Recall and F1

# ROC and AUC
## ROC(Receiver Operating Characteristic)曲线
![](/img/machine_learning/roc.png)  
如上图所示，ROC曲线的横坐标为false positive rate(FPR)，纵坐标为true positive rate(TPR)，曲线上的各个点代表着当取不同的分类阈值(当分类区的输出概率大于这个阈值时被判为正样本)时所应用的FPR和TPR，将这些(FPR,TPR)对连接起来，就得到了ROC曲线。(0,1)这个点意味着所有的样本都得到了正确的分类；(1,0)意味着所有样本都得到了错误的分类，因此ROC曲线越接近左上角，该分类器的性能越好。  
将样本按照预测输出值从大到小排序，依次选取预测值作为阈值计算(FPR,TPR)对，最后连成一条曲线。当threshold取值越多，ROC曲线越平滑。
![](/img/machine_learning/roc_plot.gif)  
可以看出每遇到一个正样本时曲线都向上延伸，每遇到一个负样本时曲线都向右延伸。

## AUC(Area Under Curve)
AUC被定义为ROC曲线下的面积，取值范围为[0.5,1]。AUC的直观意义是，它和Wilcoxon-Mann-Witney Test是等价的：**当你随机挑选一个正样本以及一个负样本时，当前的分类算法将这个正样本排在负样本前面的概率**。AUC越大，分类器的准确性越高。

## AUC计算
假设共有M个正样本，N个负样本，n=M+N。  
1. 根据AUC的定义，按预测值从大到小扫描样本直接计算阶梯下的面积。当多个正负样本的预测值相同时，ROC曲线是向斜上延伸的，此时需要计算梯形的面积。
2. 根据AUC的直观含义，可以通过遍历所有样品对，比较输出值大小确定AUC值，当正负样本的预测值相同时，按0.5计算。这样的复杂度是O(M*N)。  
3. 第二种方法的简化，按预测值从大到小排序并赋予每个样本一个rank值，预测值最大的为n最小的为1。把所有的正类样本的rank相加，减去M种两个正样本组合的情况。得到的就是所有的样本中有多少对正样本的预测值大于负类样本的预测值。对预测值相同的样本，需要赋予相同的rank值，具体的做法是把所有相等的样本的rank取平均组内重新赋值。
![](/img/machine_learning/auc_cal.png) 

# F1 vs AUC
F1分特制ROC曲线中的某个点。当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变。而Precision-Recall曲线受正负样本比例影响波动较大。



# 参考
[](http://alexkong.net/2013/06/introduction-to-auc-and-roc/)