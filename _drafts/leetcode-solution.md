---
title: Leetcode Solution
tags: [algorithm]
---
My solutions to most of Hard problems and some Medium problems on [LeetCode Online Judge](https://leetcode.com), respective code(mostly in python) is stored on [my github](https://github.com/onslaughtor/leetcode). The problems on Leetcode are easier to those in ACM-ICPC, but to raise a bug-free solution in a short time still require a lot of skills and practice. My solutions are not guaranteed to be the best(Actually, it is not necessary to raise the best solution in the interview), there are more creative idea and elegant solutions in the *Discuss*. **To get accepted is not the terminal**, keep revising and making your solutions better.  
<!-- more -->

## Recursion
### 4. Median of Two Sorted Arrays
求两个有序数组合并后的中位数。  
it's equivalent to find the kth number of the merged array. To find it, we compare the k/2th numbers of the two array and desert the smaller part, and then find the k/2th number of the sub-array recursively. Be careful when one's length is smaller than k/2.

### 22. Generate Parentheses
生成n对括号的所有合法排列。   
at each position, when there is less ')' than '(' in the previous result, then we can add a ')'; when the number of '(' is less than n, then '(' can be added. solve the next position recursively.

### 37. Sudoku Solver
数独：每行、每列、每个3*3小正方形内都不能有重复的数字。   
use three groups of set to record the number appeared in each row, column, and square and all empty grid. traverse each number from 1 to 9 and checking the sets to judge whether its legal. If legal, put the number in three respective sets and try another empty grid recursively. Remember to restore the status when the recursion tracked back.

### 87. Scramble String
用一颗二叉树表示一个字符串（分裂点随意），任意一个非叶子结点左右子节点互换称作scramble，问s1是否可以由s2通过scramble获取。   
Enumerate the split point and the check isomorphism of the respective left part and right part recursively. Remember to check scramble operation of the node itself, which means the left part of s1 can be isomorphic to either left or right part of s2. use memorization or check the the equality of sorted string to prune.     
Another solution is DP, `dp[n][i][j]` means whether the n-length substring starting from ith position in s1 and jth position in s2 is scrambled. `dp[n][i][j]`=1 when `dp[t][i][j]`=1 and `dp[n-t][i+t][j+1]`=1 for each t smaller than n, also consider the reverse isomorphism just as the recursion solution.

### 212. Word Search(Trie Tree)
给出一个字母的矩阵和一个字典，求出矩阵中包含的字典中的所有单词（四个方向）  
traverse the start point and brute force DFS is feasible for small dataset, while build a trie tree to mantain the query words is more efficient for big datasets: try the next letter if the current TrieNode has a child node of the current letter. when choosing the next node to traverse, no need to check each letter in the TrieNode(TLE because of 26x slower).

###  282. Expression Add Operators
给出一串数字和一个target，在数字之间添加运算符使结果等于target。   
Enumerate the position of the opertor and regard the left part as a number and the right part as a expression, find the candidates of the right part whose target equal to `target [+,-,*] - left` recursively. When the operator is `*`, the factor should be transmitted to the left number of the right part and keep its target unchanged. Accelerate the algorithm by memorization. 

### 329. Longest Increasing Path in a Matrix
求矩阵中最长递增序列，可以向上下左右四个方向走。

## Divide and Conquer
### 23. Merge k Sorted Lists
将k个已排序的列表合并为一个有序列表。   
similar to merge sort with a embedded merge operation for two lists

### 95/96. Unique Binary Search Trees
求1-n构成的所有的BST的个数/所有答案？  
`dp[n]=dp[n-i-1]*dp[i]` for all i in [0,n)   
Enumerate all possible number and merge it into three part: less ones, itself, and greater ones. Get the subtree of less ones and greater ones recursively and merge two subtrees into the current node.

### 241. Different Ways to Add Parentheses
给出一个包含'+','-','*'的表达式，问加括号后会有多少不同的结果。
traverse the split point and get the result of the two sides recursively. use memorization to avoid repeated computation.

### 395. Longest Substring with At Least K Repeating Characters
找出字符串的最长子串，使得子串中每个字母都至少出现了k次。


## DP
### 32. Longest Valid Parentheses
给出一个左右括号组成的序列，查找最长的合法括号序列。     
use a stack to record the position of unmatched '('. `dp[i]` represent the left end of the longest valid sequence that ends at i if i is ')'. let x be the position of the matched '(', `dp[i]=dp[x-1]` when x-1 is a ')' otherwise x. the answer is the maximum of i-dp[i]+1.

### 42. Trapping Rain Water
给出n根竖立的柱子，问可以围城多大的蓄水池。   
Let `left[i]` be the highest line left to ith line and `right[i]` be the height to the right, both are exclusive. The area above i is `min(left[i],right[i])-height[i]` if ith line is not the highest one, add each area up

### 72. Edit Distance
求两个字符串之间的编辑距离   
Let `dp[i][j]` be the edit distance between `s1[:i]` and `s2[:j]`. Then `dp[i][j]` is equal to `dp[i-1][j-1]` if `s1[i]`=`s2[j]`; otherwise the minimum of `dp[i-1][j-1]`, `dp[i][j-1] `and `dp[i-1][j]` (Replace, Insert and Delete respectively).

### 97. Interleaving String
给出字符串s1、s2，问s3是否可以由s1和s2交叉而成？  
`dp[i][j]` means whether the first i+j letters in s3 can be interleft by the fisrt i letters in s1 and first j letters in j. `dp[i][j]`=1 when `dp[i-1][j]`=1 and `s1[i]`=`s3[i+j]` or `s2[j]`=`s3[i+j]` and `dp[i][j-1]`=1.

### 115. Distinct Subsequences
字符串S中有多少个子序列是字符串T？  
`dp[i][j]` means the number of subsequence `t[:i]` in `s[:j]` when `t[i]` matches `s[j]`. `dp[i][j]`=sum(`dp[i-1][k]`) for all `k` less than `j` if `s[i]`=`t[j]` otherwise 0 and sum(`dp[-1]`) is the answer. Since `dp[i][]` only determined by `dp[i-1][]`, we can reduce it to a one-dimension array, and since the sum is accumulated, we don't need to traverse all the k. So the time complexity is O(M\*N).   
**Comment: when we swap the first dimension with the second, it doesn't work. To define the status is the first and very important process in DP.**

### 132. Palindrome Partitioning(Smart)
给一个字符串S，最少切割多少次可以使所有字串都是回文串？  
`dp[k]` means the mininum cuts for the first k characters. `dp[k]=min(dp[j]+1 for any j that s[j:k] is a palindrome)`, but the nested palindrome check is also O(n), so totally O(n\*n) for each k. However, regard kth character as the pivot of the palindrome it belongs to, we can check it while looping all j afeer it and update all `dp[j]` instead, stop the loop immediately using the monotonicity of palindrome, complexity reduced to O(n) for each k.  
**Comment: when it comes to the kth elements in DP, it not a MUST to update `dp[k]` but anyone following it(as long as there is no afteraffect)
**

### 140. Word Breaker   
给一个字符串s和一个单词的字典，求将s分割为字典中单词的方案。  
`dp[i]` record all the ending position of the word that starts at position i. `dp[k]={i}` that `s[i:k]` in the dictionary. backtrace from right to left to get the pathes(less redundant recursion than scan from left to right).  

### 188. Best Time to Buy and Sell Stock IV
给出N天股价，问最多交易K次的最大收益。   
`dp[i][j]` means the max profit when there is up to j transitions in the first i days, then `dp[i][j]`=max(`dp[i-1][j]`,`price[i]`+max(`dp[k-1][j-1]`-`price[k]`)) for all `k` less than `i`. Since the inner max loop can be updated after the ith traverse, we can loop in in the outside, update the max while the inside i loop.   
when K\*2>N, it's equal to limitless transaction and can be solved by a Greedy strategy that make a trasaction every time the prices stop going up or going down.      
**Comment: the order of updating to different dimensions affect the complexity, try to find the monotonicity and reduce repeated computation**

### 213. House Robber II
N个房间形成一个环，每个房间内有一定的财富，相邻两个房间不能同时抢劫，问最多能抢劫多少财富？   
If there is not such circle, the max profile before ith room is `dp[i]`= max(`dp[i-1]`,`dp[i-2]`+`value[i]`). Since one of the first and second has to be robbed, we can enumerate the two scenario by two similar dp where one's initial status is `dp[1]`=0, `dp[2]`=`price[2]` and another's is `dp[1]`=`dp[2]`=`price[1]`.  

### 309. Best Time to Buy and Sell Stock with Cooldown
无限次交易，两次交易之间有一天的冷冻期。   
![](http://i.imgur.com/wvR4TN8.png)   
There are three states, according to the action that you can take. we can calulate the profit of each state at time i as:
```python
s0[i] = max(s0[i-1], s2[i-1])
s1[i] = max(s1[i-1], s0[i-1] - prices[i])
s2[i] = s1[i-1] + prices[i]
```
**Comment: good idea to define the status in DP with Status Machine**

### 312. Burst Balloons
n个有数值的气球，打爆某个气球将得到`nums[left]*nums[i]*nums[right]`分，求最多可以得多少分。
Let's think about the sub problem, given a sub interval of balloons, we are only sure about the score to  burst the last balloon, so we can solve it by traversing the last burst balloon in the interval and solve the sub problem of the two sides respectively, that's  `dp[left][right]=max(dp[left][k-1]+dp[k+1][right]+num[left-1]*num[k]*num[right+1])` . To remove the after effect, we need to loop the length of the interval in the first layer, and then the start of the interval and the position of the last balloon in the inner loop.

### 368.Largest Divisible Subset
寻找元素之间可以互相整除的最大子集

### 403. Frog Jump
青蛙是否能从第一块石头跳到最后一块，如果上一步跳了k个单位则下一步可以跳[k-1,k,k+1]个单位。第一步只能跳一个单位。

### 464. Can I win
两个人轮流从[1,N]中选一个数，累加和达到M时获胜，问给定M和N先手是否可以必胜
it's a common minmax problem in game theory, and the key point is to utilize memorization to reduce the repeated computation.
The status is the numbers which are not selected and the `desiredTotal` util now. since the `desiredTotal` could be computed by the available numbers for each game, the numbers is enough to represent the status. And to traverse the available numbers and record the status efficiently, we can use bit-wise mask to record the the available numbers, instead of array or set. 

## Greedy
### 45/55. Jump Game 
给一个数组表示每个位置上最多可以向前跳多少步，至少跳多少步可以跳出尾端？  
let `dest[i]` be the farthest destination from i. Each time, we choose to jump a point x such at `dest[x]` is the farthest, in order to have more choice in the next step. Since `dest[i]` must be greater than i, the each position is just traversed once during searching the maximum. 
如果只问是否可以到达呢?   
record the farthest destination and traverse one by one until reach the end or the destination is beyond the current position(can not reach).

### 134. Gas Station
N个加油站形成环，每个加油站有`gas[i]`升油，从第i个到第i+1个加油站需要耗费`cost[i]`升油。从哪个加油站开始出发可以经过所有加油站？  
Calculate the incremental gas of each station(`gas[i]-cost[i]`), it can be proved that there must be some solution if the sum is equal or greater than 0. Start from the first station and keep moving forward until ths sum of increment is less than 0, then mark the next station as the start(the sum must be smaller starting from some station between the two ends because the sum is greater than 0 before arriving it, so any station between would be a worse choice). Repeat it until each station is visited once and there is always a answer.
An smart solution is Two Pointer in the [Discuss](https://discuss.leetcode.com/topic/5088/my-ac-is-o-1-space-o-n-running-time-solution-does-anybody-have-posted-this-solution).

### 135. Candy
每人都有一个rating值，使用最少的糖使得每人至少分到一块并且比相邻的rating低的人分到的多。  
Initially give everyone one candy and traverse from left to right, it some rating is greater than its left, give it one more than the left. Then traverse from right to left and the similar operations.

### 376. Wiggle Subsequence
一个数列两个连续元素的差正负交替则这个数列为wiggle sequence，给出一个数列，求满足wiggle sequence的最长子序列。

### 435. Non-overlapping Intervals
给出n个区间，问至少移除多少个可以使剩余的区间相互不重叠。


## Binary Search
### 33. Search in Rotated Sorted Array
在一个旋转过的有序数组里查找目标数字。   
use binary search to locate the pivot and find the target. if `nums[mid]`>=`nums[left]`, the pivot must be in the right, then compare target with the left part(because it's ascending) to determine its position; otherwise the pivot must be in the left and compare target with the right part.   
**Comment: to use the binary search ,we have to find the monotonicity even if it's not perfect**

### 153. Find Minimum in Rotated Sorted Array
在一个旋转过的有序数组（可能有重复值）中查找最小值。  
If `arr[left]`<=`arr[mid]` and `arr[right]`>`arr[mid]` then the pivot must be the right otherwise to the left. Be careful about the case when `arr[left]`=`arr[right]`.

### 240. Search a 2D Matrix II
在一个行和列都递增的矩阵中查找一个数。     
start from the top right corner, if it is greater than the target then move down otherwise move left. O(M+N).

### 287. Find the Duplicate Number
找出n+1个1到n的数字中唯一重复的一个，不能更改原数组。     
enumerate `k` by binary search and traverse the array to see whether there are at lease `k` numbers that less than or equal to `k`. If so, move to the right, otherwise move to the left. O(NlogN)

### 300. Longest Increasing Subsequence
数组中最长上升子序列。     
`last[i]` means the smallest last element in a `i`-length sequence, it is ascending with the index. So to get the longest increasing sequence ending with the current element is to find the biggest `i` that `last[i]` is less than it, which can be found by binary search. O(NlogN)

### 327. Count of Range Sum
求数列中所有和在[lower,upper]之间的子区间个数
record the sum from the first to each element of the previous array while traversing each element, check how many sums are less than `current sum -  lower`  and `current sum -  upper`  by binary search, the difference is the number of legal interval which ends with the current element.

### 354. Russian Doll Envelopes
如果一个信封的高度和宽度都严格小于另一个则可以装进去，给出N个信封问最多可以套多少个。
sort the envelopes by width increasingly and height decreasingly and then it comes to find the longest increasing subsequence of the height array. 

## Two Pointer
### 11. Container With Most Water
给出n个点，求任选两个与x轴组成的最大面积。   
Begin from the two ends, we assume the current area is W*H, to make it bigger, we must reduce W and improve H, each time desert the shorter line and shrink the segment until two ends meet.  
### 76. Minimum Window Substring
给两个字符串S和T，问S包含T中所有字符的最短字串。   
move the left pointer to scan S from left to right until all characters in T has appeared, some may appeared more than once. Then move the left pointer forward until all the characters appeared exactly once, that's one window. Move the right pointer to find other windows until the end of T.

## Array
### 15. 3Sum
给一个数组，求所有三个数相加等于0的组合。  
traverse and compute all pair of two-sum and check whether current number*-1 is in the previous two-sum set. sort the array firstly to avoid repeated answer.

### 16. 3Sum Closest
给一个数组，求任意三个数相加最接近target的和值。  
traverse the array, compute the two-sum set of the previous nums, then use binary search to find two-sum that is closet to target-current num.

### 18. 4Sum
给一个数组，求所有三个数相加等于target的组合。  
traverse the array, compute the two-sum set of the previous nums, then traverse the following nums to check whether target-current-following in the two-sum set. sort the array firstly to avoid repeated answer.

### 41. First Missing Positive 
给一个无序的数组，找出数组中缺失的第一个正整数。     
the missing Positive must between 1 to n(the length of the array). traverse the array and put the ith element to position x where `arr[i]`=`x` and x in [1,n] by swapping `a[i]` and `a[x]`. After traverse, find the first i where `a[i]`!=`i`.

### 128. Longest Consecutive Sequence
最长连续（元素）子序列。  
Traverse each element, loop and remove all the numbers consecutive to it until some is not in the Set, and that is one sequence. Since each element is removed after being visited and will not be visited after, O(n).

### 137. Single Number II
一个数组中除了某一个元素外其他元素都出现了3次。  
Sort the array and traverse each element with 3 steps, check whether `arr[i]==arr[i+1]==arr[i+2]`.

### 260. Single Number III
一个数组中两个元素出现了1次，其他元素出现了2次。   
There must be difference in some bit of the two numbers. Do exclusive operation and find the bit that equal to 1, seperate the array based on the value of this bit so that the two numbers must be seperated. Carry out exclusive operation to the sub array seperately to get two numbers.

### 229. Majority Element II
一个长度为n的数组，找出所有出现次数大于n/3次的元素。  
It's obvious there is at most two elements and it has to be in [`n/3`th smallest,`n/3`th biggest] so it's to find the `n/3`th elements and `(n-n/3+1)`th element in an array, quick sort is O(n^2) when the data is special and heapsort is stable O(nlogK).  
Another solution is to counteract, record two numbers and their frequence; when other numbers appear, counteract the frequency of the two numbers until it comes to zero and replace it with the current number.


## Stack and Queue
### 84. Largest Rectangle in Histogram(经典)
给出一个柱状图，求柱状图中最大的矩形面积。  
Scan the lines and we want to know the area of rectangle whose height is determined by this line, it is determined by the position of the first shorter line to the left and right.   
Use a stack to record the lines in increasing order, each time when we want to put a new line to the stack, while the top line is equal to higher than it, we keep popping the top element, the current line is the first shorter line left to the popped line and the previous line in the stack is the first shorter line to its left.

### 85. Maximal Rectangle
给出一个包含0和1的矩阵，求由1组成的最大子矩阵。   
similar to *Largest Rectangle in Histogram*, traverse each row and regard the continuous 1 in each column as the lines, we can get the largest rectangle blow this row.  

### 239. Sliding Window Maximum
求长度为k的滑动窗口中的最大值。
Maintain a queue ascending in index and descending in value. Pop the head if the distance to head is greater than k. Pop the tail until the tail is less than the current value and then add the current element to the tail. Then the head is the answer in this window.

### 330. Patching Array
给出一个排序数列，添加或修改数列中的数字，使得[1,n]的所有数都可以表示为数列中某些数字的和，求最小的修改次数。

### 388. Longest Absolute File Path
文件系统中最长的文件路径 

### 402. Remove K Digits
一个数字移除K位数后得到的数字最小。

### 456. 132 Pattern
求一个数列中是否存在 `i<j<k`且 `a[i]`<`a[k]`<`a[j]`。

## Heap

### 313. Super Ugly Number

## Linked List
### 25. Reverse Nodes in k-Group
给出一个列表，要求以k个元素为一组进行组内反转。  
record the tail of last group and the head of the next group, then traverse each group from right to left, hang it to the tail of the last tail one by one, lastly hang next start to the current tail. use a superhead and supertail to reduce boundary check.

### 86. Partition List
将列表中小于x的节点出现在大于x的前面，相对顺序不变。   
define small_head, small_tail, big_head, big_tail, scan the list and assert the current node to the small_tail->current->big_head or big_tail->current, update one or two of the four marks.

### 142. Linked List Cycle
检查列表中是否存在环并返回环的起点。   
Use two pointer `fast` and `low`, each time `fast` move fowards two steps and  `slow` move forwards one step. If the two pointers meet before the end, there is a cycle.
Then let `slow` points to the head and the two pointers both move one step each time and the node where they meet again is the start of the circle.  
Prove: `X`=length before the circle,`Y`=length of the circle,`K`=distance from the start of the circle to the met point. when two pointers met: `X`+m`Y`+`K`=2(`X`+n`Y`+`K`)->(m-2n)`Y`=`X`+`K`. Therefore, additional `X` steps from the met point is the start of the circle.

### 143. Reorder List
将一个列表变为L0→Ln→L1→Ln-1→L2→Ln-2→…的顺序。  
First, find the middle point with two pointer `fast` and `low`. Then reverse the list after middle node. Lastly merge the right part to the left part one by one.

### 146. LRU Cache
支持get和set操作的Least Recently Used (LRU) cache。  
use Map and Double Linked List to maitain the cache queue, add the new key to the head, remove the tail node in LinkedList and respective key in Set if it reaches the capacity. when set and get some key, the respective node should be moved to the head if it's not the head.


## Tree
### 99. Recover Binary Search Tree(Hard)

### 109. Convert Sorted List to Binary Search Tree
将一个排序的数列变成一个高度平衡的BST。   
The key point is to find the middle node of the list, initialize two node variance, each time the first moves forward one step, the second moves two steps. when the second comes to the end, the first is the middle node, solve the subtree recursively.

### 114. Flatten Binary Tree to Linked List
将一棵二分树就地变成一个列表。  
solve the subtree recursively, return the leftest and rightest node of the subtree, linked rightest node of the left subtree to node and node to the leftest node of the right subtree.

### 116/117. Populating Next Right Pointers in Each Node
让每个节点指向同层的右边节点。  
The next pointer of left child is the its right child, the next pointer the right child is it's next node's left child. Record the first node of the next level, use the next pointer and the first node of net level to traverse all nodes so that the next pointer was there when we traverse it. when it's not a perfect tree, we need another variable to record the previous node in this level.

### 124. Binary Tree Maximum Path Sum
求二分树上的最长路径。  
The maximum path of a tree is either the maximum path of its left or right subtree, or the maximum chain(starts from the root to some sub child) of left or/and right child plus itself. So find the maximun path and maximum chain, return the two variable to its father and update recursively.

### 310. Minimum Height Tree
给出一个连通无环图，则任意一个节点做根都可以构造出一棵树，找出所有可以使得树高度最小的根节点。
remove the leaf nodes layer by layer until there is less than three leaf nodes left and each of them is optional(one or two).

### 331. Verify Preorder Serialization of a Binary Tree
给出一个序列，判断是否为二叉树的合法前序遍历


### 337. House Robber III
二叉树上每个节点都有一定数额的宝贝，相邻的节点不能同时抢，最多可以抢多少宝贝？
record the maximum with and without the value itself for each node, update them from bottom recursively. 

## Regular Expression
### 10. Regular Expression Matching
判断含有'.'和'\*'的正则表达式是否完全匹配输入串。     
traverse the pattern string and use a set to record which position of the input string has been matched.  
- if the letter is '\*', pass
- if the next letter is '\*', traverse the matched position and extend each position until mismatch
- otherwise, traverse the matched position and compare whether the next position is matched to the current letter   

After traverse, if the last position of the input string in the matched set, return True.

### 44. Wildcard Matching(Hard)
'?'和'\*'的完美匹配，其中'?'代表任意字符，'\*'代表任意序列。   
a star in pattern matches all the following sequence, so we can record the matched position of last star and try to match the pattern with all the subsequence after this position until meet another star or entirely matched.     
A faster way is to use KMP to find the first matched pattern until next star.   
DP is a more straightforward solution but got TLE. let `dp[i][j]` represent whether first i letters in pattern matched first j letters in string, `dp[i][j] = dp[i-1][j-1]` when `pattern[i]` matches `string[j]` and `dp[i][j]=dp[i][j-1] | p[i-1][j]` if pattern[i] is '\*' otherwise 0. A rolling array to can reduce the memory complexity to O(n).

## Others
### 3. Longest Substring Without Repeating Characters：
求没有重复字符出现的最长子串。   
traverse the string, record the last position of each letter and the position last repeated letter. 
if the current letter has appeared before, the beginning of the qualified substring ending with current letter is minnum of lastAppear and lastReapt otherwise the lastReapeat. Update the two kinds of variable and maximum length.

### 30. Substring with Concatenation of All Words
给出一个字符串和一组长度相同的单词，求包含所有单词的字串。     
the length of the matched substring must be constant. enumerate the starting point(0,len(word)-1) and traverse from left to right, move len(words) each time and update the number of matched words by add one and remove one.

### 31.  Next Permutation
traverse from right to left to find the first ascending pair. reverse the right part(descending sequence), swap the last one on the left with the smallest bigger one on the right.

### 126/127.  Word Ladder
给出一个begin_word和一个end_word以及一个单词列表，问从begin_word转化到end_word的最短序列。      If two words are different in just one letter then we assume they are connected, the question is to find the shortest path in the undirected graph. Since the length of each edge is 1, we can ultilize BFS to find the shortest path. At each word, there is no need to traverse all the word to find its neighbors, while to traverse index and 26 letters is more efficient, reduce the complixity from O(N\*N\*L) to O(N\*L\*26).   
To record all pathes, we need to another array to record the level of each node and record the next node on the path based on their level. Then use backtrack of this array to output the pathes. 

### 149.  Max Points on a Line 
二维坐标系里最多有多少个点共线？  
Traverse every point and count the number of gradient to all other lines. Careful about the verticle lines(infinity gradient) and repeated points(no gradient). O(n^2).

### 166.  Fraction to Recurring Decimal
将两个整数相除表示为（循环）小数的形式。   
deminish the numerator bit by bit until some numerator has been appeared, record the position of each numerator and append the '()' between the two positions.

### 214. Shortest Palindrome(经典)
给一个字符串添加最少的字符使之变为回文串。   
the value of `next` array of the string `S`+'#'+reverse(`S`) in KMP

### 335. Self Crossing
从原点出发后一次向东、西、南、北走若干步，问路径是否会否交叉。

### 336. Palindrome Pairs
给出一个长度为N的单词列表，问有哪些单词对拼接起来可以组成一个回文串。
record the reversed string of each word and then traverse each word's substring, if the substring(both left part and right part) is in the reversed strings and the other part itself is palindrome, then the word can form a palindrome with the respective word of the reversed string, the time complexity is O(N\*K\*K) where K is the length of the word.

### 352. Data Stream as Disjoint Intervals

给出一个整数的数据量，请求时以不重叠区间的形式返回之前的数据
utilize TreeMap in Java to record the intervals where the key is the left end of the interval and value is the right end, when a new number comes, check the `ceilingKey()` and `floorKey` to get its neighbor intervals and  decide whether combine two adjacent intervals, add the number  to the one interval or generate a new interval.

### 397. Integer Replacement
给一个整数n，如果是偶数可以变为n/2,否则可以变为n-1或n+1，问n至少经过多少次变换可以变为1。



