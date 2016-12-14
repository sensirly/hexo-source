---
title: Data Visualization (UIUC)
tags:
  - online course
  - data science
date: 2016-06-22 21:23:03
---
# week 1: The Computer and the Human
visualization is an interface between the computer and the human.    因此我们要首先了解human和computer的特点才能做好visualization。

## Photorealism
用2D画面表示3D效果常用的技巧
- occlusion：不透明物体的覆盖关系暗示远近关系（strongest cue）
- illumination：通过亮度差异暗示平面方向，强调细节
- shadowing：制造光线的occlusion
- perspective：大小暗示深度（近大远小）
![](/img/coursera/photorealism.JPG)
<!-- more -->
## Fitts’ Law（费茨法则）
当一个人用鼠标来移动鼠标指针时，屏幕上的目标的某些特征会使得点击变得轻松或者困难。目标离的越远，到达就越是费劲。目标越小，就越难点中。
![](/img/coursera/fitt-law.JPG) 
**Fitts' Law鼓励减少距离，增加目标大小以提升用户效率**。比如Mac OS将Dock放在最底端、windows开始菜单在左下角。这些区域都是可以被“无限可选中的”，只需要大幅度晃动鼠标就能到达目标区域，相当于增大了目标大小；而右键菜单随时可以触发而不需要将鼠标移向程序主菜单，相当于减少距离。Mac系统中的交互设计更好的应用了费茨法则，因此操作效率更高。  
![](/img/coursera/os_margin.png)
更多应用案例参照：[设计法则： Fitts’ Law / 菲茨定律（费茨法则）](http://www.jianshu.com/p/36b610bac7a2)

## Human Retina
- 对亮度比对颜色更敏感。因为亮度和颜色是分别传输的，感应亮度的神经元数量更大
- chromatic aberration（色差）：因为不同波长的色光有不同的折射率，透镜（人的眼睛）无法将各种波长的色光都聚焦在同一点上。蓝色的波长比红和绿更容易偏离视网膜，使图像看起来模糊，因此应该避免纯蓝色的文本。**暖色调更容易聚焦**。
- Color Perception:Luminance=31%R+59%G+10%B。Yellow=Green+Red，因此黄色非常亮

## Lateral Inhibition(Perceiving 2D) 
we see things in context because of these local comparisons that our perceptual system does. 神经系统在处理图像时会放大差异，已便于识别
![color context](/img/coursera/color_context.JPG)
相同的紫色因为环境色的不同亮度发生了改变
![orientation context](/img/coursera/orientation_context.JPG)
左边橙色梯形的斜率是2/5，右边的斜率是5/13。
![size context](/img/coursera/size_context.JPG)
我们看到的颜色、形状、大小会因为跟环境的对比发生扭曲,因此这些特点有时也会干扰我们正确解读数据。

# week 2:  Visualization of Numerical Data

## Mapping and Chart
数据根据不同维度可分为：连续的-离散的，有序的-无序的。不同类型的数据需要借助不同的图像特征来做map，比如对于数值，位置、长度、角度等一维的特征最明显。
![data mapping](/img/coursera/data_mapping.JPG)
在选取图表表达数值时，条形图更合适一些，因为它利用了位置和长度，而线形图是利用了位置而没有使用长度。

## High Dimension
高维度展示不易辨别，因此常使用下面的技巧用低维空间表达高维数据，但也经常会引起误导
### Glyphs（符号）
![Glyphs（符号）](/img/coursera/glyphs.JPG)
在图标的形状之上增加一些符号提供额外的信息，比如用颜色表示的热力图，这些特征虽然不是表达能力最强的特征，但是也起到了很好的辅助作用
### Parallel Coordinates
![parallel coordinate](/img/coursera/parallel_coordition.JPG)
平行坐标系可以展示高维数据，但是只有相邻两个维度的相关性可以直观的被表达，因此需要人工的设置顺序。
### Stacked Graphs
![stacked graph](/img/coursera/stacked_graph_order.JPG)
- Central Limit Thereo:当更多的bar被加入时，整体的变化会减缓
- 不同bar的相互顺序也会影响趋势的视觉感受（Position>Length）
- 线形图的stacked graph比柱状图的更平缓
- 可以更改baseline使图像变化更加平缓

# 感受
非常悲剧，因为这个课有点boring所以一直没跟上节奏，等我有闲情逸致想继续刷的时候发现已经close了。。。总体感觉一般吧，节奏太慢，实用性不是很强，因此就到此为止吧，不打算继续追了。但是里面提到的一些交互设计原理还是挺有趣的，改天有机会直接去研究交互原理了。 
感觉coursera上的好课就那么多（比如Stanford系列），在这个已经开始四处收钱的模式下，coursera的教学效率已经大不如前了，所以打算告别coursera一段时间，winter is coming，抓紧时间去做点更高效的事情。