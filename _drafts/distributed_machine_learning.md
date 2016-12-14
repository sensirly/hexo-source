---
title: 分布式机器学习平台
tags: [machine learning]
---

# Hadoop
Hadoop主要由两部分组成，HDFS和MapReduce。HDFS通过将块保存多个副本的办法解决了大数据可靠存储问题；MapReduce借鉴函数式编程语言的思想，通过Mapper和Reducer的抽象提供一个可以在多台机器上并发执行的编程模型，复杂的数据处理通过分解为由多个Mapper和Reducer组成的有向DAG依次调度得到结果。如下所示的wordcounter是一个经典的mapreduce编程示例。   
![](img/machine_learning/distributed_system/MapReduceWordCount.png)   

## 可靠性
MapReduce通过把对数据集的大规模操作分发给网络上的每个节点实现可靠性；每个节点会周期性的把完成的工作和状态的更新报告回来。如果一个节点保持沉默超过一个预设的时间间隔，主节点录下这个节点状态为死亡，并把分配给这个节点的数据发到别的节点重新运行。

## 缺点
- 只提供MapReduce一种计算框架，不利于算法的灵活实现。比如做Join等简单操作时需要通过很绕的方式实现
- 每个Mapper的结果保存在buffer中（如果buffer空间不足也会写到磁盘中，成为shuffle spill files），Reducer的结果都要保存到磁盘上。频繁读写，不利于迭代式算法的高效实现
- Reducer要等待所有的Mapper执行完才开始运行，数据处理延时长，不利于处理流式数据

# Spark
Spark可以看做是针对Hadoop缺点进行的优化，运行在现有的Hadoop分布式文件系统(HDFS)基础之上提供额外的增强功能，通过内存数据共享大大提高了迭代效率。通常来讲，针对数据处理有几种常见模型，包括：Iterative Algorithms，Relational Queries，MapReduce，Stream Processing，例如Hadoop MapReduce采用了MapReduces模型，Storm则采用了Stream Processing模型。Spark通过RDD(Resilient Distributed Datasets)混合了这四种模型。

RDD是一种分布式只读分区集合的数据结构，Spark将数据存储在不同分区上的RDD之中，RDD可以帮助重新安排计算并优化数据处理过程，且具有容错性（即当某个节点或任务失败时RDD会在余下的节点上自动重建）。RDD支持两种类型的操作：
- 变换（Transformation）：调用一个变换方法，返回一个新的RDD集合，比如map，filter，flatMap，groupByKey，reduceByKey等。
- 行动（Action）：计算全部的数据处理查询并返回结果值。比如reduce，collect，count等

Spark对于有向无环图Job进行调度，确定阶段（Stage），分区（Partition），流水线（Pipeline），任务（Task）和缓存（Cache），进行优化，并在Spark集群上运行Job。RDD之间的依赖分为宽依赖（依赖多个分区）和窄依赖（只依赖一个分区），在确定阶段时，需要根据宽依赖划分阶段。根据分区划分任务。   
![](img/machine_learning/distributed_system/spark_stage.png)    

# MPI
- no fault tolerance

# Parameter Server

# Giraph

[与 Hadoop 对比，如何看待 Spark 技术？](http://www.zhihu.com/question/26568496)
[知乎：写分布式机器学习算法，哪种编程接口比较好？](https://www.zhihu.com/question/22544716)
[“分布式机器学习的故事”系列分享](http://cxwangyi.github.io/notes/2014-01-20-distributed-machine-learning.html)