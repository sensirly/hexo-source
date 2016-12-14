---
title: python正则表达式
tags:
  - python
  - coursera
date: 2016-05-26 19:57:27
---
# 常用模式
| 模式 | 含义 |
| :---:| :--------|
| ^ | Matches the beginning of a line |
| $	| Matches the end of the line |
| .	| Matches any character |
| \d| Match one digit				|
| \D| Matches any non-digit character |
| \w| Match one number or one digit   |  
| \s| Matches whitespace |
| \S| Matches any non-whitespace character |
| ? | Repeats a character zero or one times |
| *	| Repeats a character zero or more times |
| +	| Repeats a character one or more times |
| [aeiou] | Matches a single character in the listed set |
| [^XYZ] | Matches a single character not in the listed set |
| [a-z0-9] | The set of characters can include a range |
| () | Indicates where string extraction is to start and to end |
<!-- more -->
Lazy means match shortest possible string.
Greedy means match longest possible string.(default)
For example, the greedy `h.+l` matches 'hell' in 'hello' but the lazy `h.+?l` matches 'hel'.

# python调用方式
`import re`引入模块直接开始使用正则匹配各种功能
- re.search() 返回True\False
- re.match() 匹配起始位置成功返回起始位置，否则返回non
- re.findall()  返回所有匹配的list
另外一种使用方法
1. 先将正则表达式的字符串形式编译为Pattern实例
1. 使用Pattern实例处理文本并获得匹配结果（一个Match实例）
1. 使用Match实例获得信息
两种方法是等价的，只不过第二种支持pattern的复用


# 正则匹配复杂度
Python正则匹配使用基于回溯的一种NFA实现。通过数据比较，在最坏的情况下用Thompson NFA实现的awk表现比匹配回溯的NFA要好很多倍。最坏情况下的复杂度不一样，回溯NFA是O(2^N)，而Thompson的复杂度是O(N^2)。参见[正则表达式匹配和NFA/DFA](http://cyukang.com/2014/01/04/regular-expression-matching-dfa.html)

# practice
`print re.findAll([0-9]+,'My favorite 2 number are 19 and 42')`    
['2','19','42']

`x='From stephen.marq@uct.ac.za to sansa@uci.edu Sat Jan 5 09:04:15 2008' `   
`y=re.findall('\S+@\S+',x) `  
['stephen.marq@uct.ac.za','sansa@uci.edu']   
`y=re.findall('From (\S+@\S+)',x)`   
['stephen.marq@uct.ac.za']  
  

# Reference
[课程链接](https://www.coursera.org/learn/python-network-data/home/week/2)
[regular expression验证工具RegExr](http://regexr.com/)

