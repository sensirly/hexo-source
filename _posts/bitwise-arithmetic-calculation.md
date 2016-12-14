---
title: 位运算实现加减乘除
tags:
  - java
  - algorithm
date: 2016-06-14 19:00:52
---

## 加法
一位加法时，和是`x XOR y`的结果,而进位恰好是`x AND y`的结果.
多位加法时，可将x, y的每一位级联计算：先计算x和y的第零位，将计算所得的进位传入到x和y的第一位的计算中，依次进行直到计算完最高位为止，此时将每一位计算所得的和连接起来就是最终的和。计算机中的加法也是使用这种原理来实现的。
然而，逐位取出计算操作复杂，可以一次求出所有位上的和及进位，然后递归调用此方法直至进位为0。
```java
public int add(int x, int y) {
//      if(y==0)
//          return x;
//      return add(x ^ y,(x & y) << 1);
    while (y != 0) {
        int tmp = x ^ y;
        y = (x & y) << 1;
        x = tmp;
    }
    return x;
}
```
<!-- more -->

## 减法
减法是加法的逆运算，加法中的可以从低位开始依次往上传递进位，而且高位对低位的计算不产生影响。而减法则需要从高位获得借位，高位会对低位的计算产生影响。如果是小数字减大数字则计算过程更复杂，因此直接实现减法对计算机来说很复杂，而且效率很低。因此一般通过转化成相反数的加法完成，而只要想得到一个数的相反数，只要对这个数求2-补码就可以了，即取反加1操作。
```java
public int subtract(int x, int y) {
    return add(x, add(~y, 1));
}
```

## 乘法
根据乘法演算的过程，
- 根据乘数每一位为1还是为0，决定相加数取被乘数移位后的值还是取0
- 各相加数从乘数的最低位开始求值，并逐次将相加数(被乘数)左移一位，然后求和
- 符号位根据同号为正异号为负的原则
```java
/**
          1 0 1 1
*         1 1 0 1
 ------------------
          1 1 0 1
        1 1 0 1 0
      0 0 0 0 0 0
    1 1 0 1 0 0 0
 ------------------
  1 0 0 0 1 1 1 1
**/
public int multiply(int x, int y) {
    int ans = 0;
    int sign = 0;
    if (x < 0) {
        sign = ~sign;
        x = add(~x, 1);
    }
    if (y < 0) {
        sign = ~sign;
        y = add(~y, 1);
    }
    while (y > 0) {
        if ((y & 1) > 0)
            ans += x;
        x <<= 1;
        y >>= 1;
    }
    if(sign>0)
        return add(~ans, 1);
    else
        return ans;
}
```

## 除法
除法就是由乘法的过程逆推，依次减掉（如果x够减的）y^(2^31),y^(2^30),...y^8,y^4,y^2,y^1。减掉相应数量的y就在结果加上相应的数量。
```java
public int divide(int x, int y) {
    int ans = 0;
    int sign = 0;
    if (x < 0) {
        sign = ~sign;
        x = add(~x, 1);
    }
    if (y < 0) {
        sign = ~sign;
        y = add(~y, 1);
    }
    for (int i = 31; i >= 0; i--) {
        if (x >= (y<<i)) {
            x -= y<<i;
            ans+= 1 << i;
        }
    }
    if(sign>0)
        return add(~ans, 1);
    else
        return ans;
    return ans;
}
```
