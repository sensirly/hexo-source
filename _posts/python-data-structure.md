---
title: python数据结构
tags:
  - python
  - algorithm
date: 2016-08-15 19:00:00
---

记录一下使用python数据结构过程中踩过的坑及一些常见应用。同时参照了[《Data Structures and Algorithms in Python》](http://multimedia.ucc.ie/Public/training/cycle1/algorithms-in-python.pdf)这本书，其中讲了python数据结构底层的实现原理及很多高效使用的建议。

# List
列表是一个可变的序列，可以直接按索引访问，也可以动态的删减,表示为`[x,y,z]`
## 多维数组
如果使用`array = [[0] * 3] * 3`对一个3*3的二维数组进行初始化，当操作`array[0][1] = 1`时，发现整个第二列都被赋值，变成:`[[0,1,0],[0,1,0],[0,1,0]]`   
[The Python Standard Library](http://docs.python.org/library/index.html)里的解释是：list * n—>n shallow copies of list concatenated, n个list的浅拷贝的连接。
因此正确的初始化方式应该是：  
`array=[([0] * 3) for i in range(3)]`
<!-- more -->
## 排序
- 使用容器自带的排序函数`list.sort()`，此时原列表结构被改变;  
- 使用python的内建函数`sorted(list)`，原列表结构不发生改变，返回一个新的列表。

两个函数的参数类似，可以通过指定key或者比较函数的方式进行排序：  
`sorted(iterable,[cmp=func()],[Key=?],[reverse=True/False])`   
其中reverse在不指定时默认为False。当列表中元素较为复杂时，常使用lambda函数指定key和cmp函数，参照[知乎:Lambda 表达式有何用处？如何使用？](https://www.zhihu.com/question/20125256)。   
```python
from operator import itemgetter

l1 = [{'name':'abc','age':20},{'name':'def','age':30},{'name':'ghi','age':25}]
print sorted(l1,key = lambda x:x['age'],reverse=True)

students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10),]
print sorted(students, cmp=lambda x,y : cmp(x[2], y[2]))
print sorted(students, key=itemgetter(2))  #使用operator指定key，与上面指定cmp效果等价
print sorted(students, key=itemgetter(1,2)) #使用operator进行多级排序
'''
[{'age': 30, 'name': 'def'}, {'age': 25, 'name': 'ghi'}, {'age': 20, 'name': 'abc'}]
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
[('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]
'''
```

## 有序数组操作 bisect
`bisect`中的插入和查询操作都是基于二分查找实现的，单次操作复杂度为O(logn)，因此操作之前必须保证数组是有序的。如果插入的是复杂对象而不是数字，可以插入元组(val,obj)实现，这样就会自动根据元组的第一个元素进行排序。
```python
import bisect
a=[1,3,5,7,9]
#如果元素存在返回索引，如果不存在返回后继的索引（如果将x插入数列后的位置）
print bisect.bisect(a,2) #2
print bisect.bisect(a,3) #2
print bisect.bisect_left(a,3) #1  总是返回大于等于这个数的索引
# 向有序数组插入新元素
bisect.insort(a,4) # [1,3,3,4,5,7,9]
bisect.insort(a,3) # [1,3,3,5,7,9]
```

## 常用操作
```python
# 按索引插入删除
l=[1,2,3,5,7]
l.insert(1,5) #[1,5,2,3,5,7]
l.pop(2)#[1,5,3,5,7]
# 查询索引
print l.index(5) #1
# 查询元素个数
print l.count(5) #2
# 移除某个值
l.remove(5) #[1,3,5,7]
l.reverse() #[7,5,3,1]
```
## 实现栈和队列
列表对象支持类似的操作，但只是优化了固定长度操作的速度。像pop(0)和insert(0,v)这些改变列表长度和元素展示位置的操作都会带来 O (n)的内存移动消耗。`deque`是一种由队列结构扩展而来的双端队列。无论从队列的哪端入队和出队，性能都能够接近于O(1)。 
```python
# using List as Stack  
l.append(9)#[1,3,5,7,9]
l.append(11)#[1,3,5,7,9,11]
l.pop()#[1,3,5,7,9]
# using deque as Queue
from collections import deque
q=deque([2,3,4])
q.append(5) #deque([2,3,4,5])
q.appendleft(1) #deque([1,2,3,4,5])
print q[0] #1
print q.popleft() #deque([2,3,4,5])
```

## 优先队列（堆）
堆（小顶堆min-heap）用树形结构维护数列，使得heap[k] <= heap[2*k+1] and heap[k] <= heap[2*k+2]，max-heap性质相反。pyhton中实现的是小顶堆。 
```python
import heapq
queue=[3,7,4,1,5]
heapq.heapify(queue)
print queue#[1,3,4,5,7]
print heapq.heappop(queue) #1
print queue#[3,5,4,7]
heapq.heappush(queue,6)
print queue#[3,5,4,7,6]
heapq.heapreplace(queue,1)# more efficient than a heappop() followed by heappush()
print queue#[1,5,4,7,6]
heapq.heappushpop(queue,2)# more efficiently than heappush() followed by heappop()
print queue#[2,5,4,7,6]
```
如果插入的是复杂对象而不是数字，可以插入元组(val,obj)实现，这样就会自动根据元组的第一个元素进行排序


# 元组(tuple)
元组与列表类似，但是一个不可修改的序列，表示为`(x,y,z)`


# 字符串
## 列表转换
通过list()构造函数或者split()方法将字符串转换为列表。通过join()函数将列表转换为字符串。
```python
s='abcd'
l=list(s)
ss=''.join(l)
words=sentence.split(' ')
``` 
直接使用`+`做字符串拼接的复杂度是O(n*n),先将字符append到数组中在转化为字符串可将复杂度降低到O(n)

## 字母遍历
```python
import string
for word in string.uppercase:
    print word
for word in string.lowercase:
    print word
``` 
Python提供了`ord()`和`chr()`两个内置的函数，用于字符与ASCII码之间的转换
```python
print ord('a') 
print chr(97) 
```


# 序列通用操作
列表、元组、字符串都属于序列，支持通用的序列操作：

+ 索引：索引从0（从左向右）开始，也可以从最后一个位置（从右向左）开始，编号是-1。复杂度O(1)
+ 分片：通过s[x:y:z]访问序列的左x开右y闭区间,z表示步长。当某个值缺失表示端点，如s[::-1]表示序列反转；当x大于y时返回空序列。复杂度O(y-x)
+ 序列相加：完成序列拼接，但只支持统一类型的序列之间相加。复杂度O(n2)
+ 序列乘法：实现序列的复制。`s='abc',s*3='abcabcabc'`。杂度O(n*k)
+ 检查成员：`'x' in s`，返回True或False。复杂度O(n)
+ 长度、最大、最小值: len(s), 复杂度O(1)；max(s), min(s)，复杂度O(n)


# 字典(dict)
+ 创建方式：`d={'a':1, 'b':2}` 或 `d={}` 或 `d=dict()`
+ 字典的键可以是任何不可变类型（数字、字符串、元组）;
+ 当键不存在时可以自动创建，不需要先检查是否存在再赋值；
+ 删除key为x的记录`del dict['x']`；清空词典所有条目`dict.clear()`    
+ 同时遍历关键字和对应的值 `for k,v in d.items()`

## OrderedDict 
OrderedDict只是按照插入顺序进行了排序，并没有按照key进行排序，无法实现类似于java中treemap的功能


# 集合(set)
集合就是由序列构建的，表示为`set([x,y,z])`

+ 并集 `a.union(b)``a|b`
+ 交集 `a&b`
+ 差集 `a-b`
+ update方法：把要传入的元素拆分，做为个体传入到集合中

# 参考
[Stack Overflow上热门python问题翻译](https://taizilongxu.gitbooks.io/stackoverflow-about-python/content/index.html)
[《Data Structures and Algorithms in Python》](http://multimedia.ucc.ie/Public/training/cycle1/algorithms-in-python.pdf)
[Python3 数据结构|菜鸟教程](http://www.runoob.com/python3/python3-data-structure.html)
[Python常见数据结构整理](http://www.cnblogs.com/jeffwongishandsome/archive/2012/08/05/2623660.html)
[Python中的高级数据结构](http://blog.jobbole.com/65218/)