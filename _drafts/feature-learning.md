
机器学习算法的成功取决于数据表达（特征）的选择，现有的一些特征工程依赖人类的智慧和先验知识弥补了特征表达方面的缺失，但是比较低效。在概率模型中，一个好的特征能够捕捉数据的先验分布，同时能够成为预测器的有效输入。

## WHAT MAKES A REPRESENTATION GOOD
- Smoothness and the Curse of Dimensionality: 在图像和NLP领域，必须依赖非线性的特征才能表达复杂的数据结构。一些基于kernel的算法依赖于local generalization：通过相邻样本的平滑消除预测函数的波动。但这些算法会面临维度灾难，因此可以将smoothness-based算法用于这些表达的顶层。
- Distributed representations：特征应该具有很好的表达性（expressive），衡量特征表达性强弱的指标是：需要多少的参数可以区分固定的输入空间。比如线性模型是O(N)的，神经网络的参数量是O(2^k)的，后者是更加distributed(sparse)的表达。
- Depth and abstraction：深度加购有两个优点，一是可以重复利用distributed representation(weight sharing?) ，二是可以在深层表达更加抽象的特征。
- Disentangling Factors of Variation：数据可能有多个来源，消除特征来源的不确定性
- Good criteria for learning representations：特征表达学习很难定义损失函数

## Building Deep Representation
**greedy layerwise unsupervised pre-training**可以逐层学习特征的结构，使用非监督学习在每层学到一组变换，然后之前的层的变换进行融合进而初始化深层网络。同时这个方法的实验表明，逐层的提取特征能够获取更好的特征表达。也有一些使用监督学习进行pre-trained的工作，但是效果不如非监督学习好。几种构建方法：
- stack pre-trained RBMs into a Deep Belief Network，第一层使用RBM，后面使用sigmoid belief network。
- combine the RBM parameters into a Deep Boltzmann Machine(DBM),DBM可以通过最大似然估计训练获取，这种joint train可以带来额外的收益。
- stack RBMs or autoencoders into a deep auto-encoder：