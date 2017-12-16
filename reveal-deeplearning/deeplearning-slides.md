---
presentation:
  width: 1280
  height: 800
  font-family: 微软雅黑
  enableSpeakerNotes: true
---

<!-- slide -->
# 深度学习入门
> East196

[GitHub：http://github.com/east196](http://github.com/east196)

<!-- slide -->
## 深度学习
### 是什么？
机器学习的最前沿分支
深度学习 = 深度神经网络
<!-- slide vertical=true-->
### 能做什么？
预测未来
```dot
digraph d3 {
预测未来
预测未来->分类
预测未来->回归
预测未来->生成
}
```

<!-- slide vertical=true-->
### Why？
简单粗暴效果好！
#### 1. 能Work
- 越来越大的数据
- 越来越快的电脑
- 越来越多的优化技巧

#### 2. 能解决问题

越来越多的Neural Network模型

<!-- slide vertical=true-->
### AI？
AI目前用途最广泛的分支，疯狂融合其他分支的内容
AlphaGo ~~~ DNN+蒙特卡洛搜索树
自动驾驶 ~~~ CNN
聊天机器人 ~~~ RNN
生成文本，图片，语音，视频 ~~~ GAN
自动打游戏 ~~~ DQN



<!-- slide -->
## 机器学习的分类
- 监督学习：分类，回归
- 无监督学习：聚类
- 强化学习：奖惩机制

<!-- slide vertical=true-->
### 深度学习遍及机器学习的方方面面
- 监督学习
- 无监督学习
- 强化学习
### 但最强的还是在于
**监督学习**
- 分类
- 回归

<!-- slide -->
## 从y=wx+b谈起
$y=f(x)$
## 简单的函数
二元一次方程 - 普通的线性关系
$y=wx+b$

<!-- slide vertical=true-->
 给两组数据：
```
10，2
3，4
```

构成方程：
$2 = 10w+ b$
$4 = 3w+b$

怎么解？：）
<!-- slide data-background-image="assets/sklearn.jpg" vertical=true -->


<!-- slide vertical=true-->
```python {cmd=true matplotlib=true}

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
#指定默认字体  
matplotlib.rcParams['font.family']='simhei'  
#解决负号'-'显示为方块的问题  
matplotlib.rcParams['axes.unicode_minus']=False  

#解方程 y = wx + b
x = np.array([10,3])
y = np.array([2,4])

A = np.vstack([x, np.ones(len(x))]).T
w, b = np.linalg.lstsq(A, y)[0]
#print(w, b)

# 再来画个图
plt.axis([0, 15, 0 ,6])
plt.plot(x, y, 'o', label=u'原始数据', markersize=10)

t = np.linspace(-10,20,10)
plt.plot(t, w*t + b, 'r', label=u'线性方程')
plt.legend()
plt.show()
```


<!-- slide vertical=true-->
然而，现实是：
```python {cmd=true matplotlib=true}

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
#指定默认字体  
matplotlib.rcParams['font.family']='simhei'  
#解决负号'-'显示为方块的问题  
matplotlib.rcParams['axes.unicode_minus']=False  

# 模拟真实数据
x = np.linspace(-15,20,100)
y = 10*x +np.random.rand(100)*120
z = 3*x*x +np.random.rand(100)*160
m = 2*x*x +10*x +np.random.rand(100)*250

# 再来画个图
plt.plot(x, y, 'o', label=u'真实数据', markersize=10)
plt.plot(x, z, 'x', label=u'数据', markersize=10)
plt.plot(x, m, '*', label=u'数据', markersize=10)

plt.legend()
plt.show()
```
<!-- slide vertical=true-->
## 传统的机器学习方法
use `scikit-learn`
分类，回归，聚类，etc


<!-- slide data-background-image="assets/sklearn.jpg" vertical=true -->


<!-- slide vertical=true-->
## 最简单的神经网络
```dot
digraph Ped_Lion_Share           {
rankdir=LR;
label = "最简单的神经网络" ;
"x" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"f" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"y" [shape=circle  , regular=1,style=filled,fillcolor=white   ]
"x"->"f"
"f"->"y"
}

```
咦，效果一般啊？
一定是打开的方式不对！

<!-- slide vertical=true-->
## 权重Weight与偏置Biase
$ y = wx+b $
面熟对不对？
求解线性问题
权重和偏置怎么设置？
我也不知道，那就随机吧...

<!-- slide vertical=true-->
## 激活函数
面对现实
非线性世界

<!-- slide vertical=true-->
激活函数 Sigmoid&Tanh
```python {cmd=true matplotlib=true}
import math  
import matplotlib.pyplot as plt  
import numpy as np  
import matplotlib as mpl  
mpl.rcParams['axes.unicode_minus']=False  


def  sigmoid(x):  
    return 1.0 / (1.0 + np.exp(-x))  

fig = plt.figure(figsize=(6,4))  
ax = fig.add_subplot(111)  

x = np.linspace(-10, 10)  
y = sigmoid(x)  
tanh = 2*sigmoid(2*x) - 1  

plt.xlim(-11,11)  
plt.ylim(-1.1,1.1)  

ax.spines['top'].set_color('none')  
ax.spines['right'].set_color('none')  

ax.xaxis.set_ticks_position('bottom')  
ax.spines['bottom'].set_position(('data',0))  
ax.set_xticks([-10,-5,0,5,10])  
ax.yaxis.set_ticks_position('left')  
ax.spines['left'].set_position(('data',0))  
ax.set_yticks([-1,-0.5,0.5,1])  

plt.plot(x,y,label="Sigmoid",color = "blue")  
plt.plot(2*x,tanh,label="Tanh", color = "red")  
plt.legend()  
plt.show()  
```
<!-- slide vertical=true-->
激活函数 ReLU
```python {cmd=true matplotlib=true}
import math  
import matplotlib.pyplot as plt  
import numpy as np  
import matplotlib as mpl  
mpl.rcParams['axes.unicode_minus']=False  

fig = plt.figure(figsize=(6,4))  
ax = fig.add_subplot(111)  

x = np.arange(-10, 10)  
y = np.where(x<0,0,x)  

plt.xlim(-11,11)  
plt.ylim(-11,11)  

ax.spines['top'].set_color('none')  
ax.spines['right'].set_color('none')  

ax.xaxis.set_ticks_position('bottom')  
ax.spines['bottom'].set_position(('data',0))  
ax.set_xticks([-10,-5,0,5,10])  
ax.yaxis.set_ticks_position('left')  
ax.spines['left'].set_position(('data',0))  
ax.set_yticks([-10,-5,5,10])  

plt.plot(x,y,label="ReLU",color = "blue")  
plt.legend()  
plt.show()  
```
<!-- slide vertical=true-->
### BP神经网络
回头修正参数

<!-- slide vertical=true-->
### 可是，我要识别一张图片，我该输入什么？
输入一般称之为特征
特征要靠自己分析，然后选取
比如我要识别车牌上的一个字

画个九宫格，9个特征
上中下，3个特征
整图，1个特征

9+3+1=13

<!-- slide vertical=true-->
NO！
现在计算机跑这么快了，
我要把 宽*高*RGBA 直接扔进去！！！

<!-- slide -->
## DNN
更大更深的神经网络

<!-- slide -->
NN打油诗
    - East196

机器性能大提升，
海量数据在产生。
群策群力来贡献，
神经网络大又深。

<!-- slide vertical=true-->
### 自动编码器
```dot
digraph d3 {
rankdir=LR;


label = "自动编码器" ;

"x0" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x1" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x2" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x3" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x4" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x5" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x6" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x7" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x8" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"x9" [shape=circle     , regular=1,style=filled,fillcolor=white   ]


"e1" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"e2" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"e3" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"e4" [shape=circle     , regular=1,style=filled,fillcolor=white   ]

"m1" [shape=circle  , regular=1,style=filled,fillcolor=white   ]
"m2" [shape=circle  , regular=1,style=filled,fillcolor=white   ]

"d1" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"d2" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"d3" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"d4" [shape=circle     , regular=1,style=filled,fillcolor=white   ]

"r0" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r1" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r2" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r3" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r4" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r5" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r6" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r7" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r8" [shape=circle     , regular=1,style=filled,fillcolor=white   ]
"r9" [shape=circle     , regular=1,style=filled,fillcolor=white   ]

"x0"->"e1"
"x1"->"e1"
"x2"->"e1"
"x3"->"e1"
"x4"->"e1"
"x5"->"e1"
"x6"->"e1"
"x7"->"e1"
"x8"->"e1"
"x9"->"e1"
"x0"->"e2"
"x1"->"e2"
"x2"->"e2"
"x3"->"e2"
"x4"->"e2"
"x5"->"e2"
"x6"->"e2"
"x7"->"e2"
"x8"->"e2"
"x9"->"e2"
"x0"->"e3"
"x1"->"e3"
"x2"->"e3"
"x3"->"e3"
"x4"->"e3"
"x5"->"e3"
"x6"->"e3"
"x7"->"e3"
"x8"->"e3"
"x9"->"e3"
"x0"->"e4"
"x1"->"e4"
"x2"->"e4"
"x3"->"e4"
"x4"->"e4"
"x5"->"e4"
"x6"->"e4"
"x7"->"e4"
"x8"->"e4"
"x9"->"e4"
"e1"->"m1"
"e2"->"m1"
"e3"->"m1"
"e4"->"m1"
"e1"->"m2"
"e2"->"m2"
"e3"->"m2"
"e4"->"m2"
"m1"->"d1"
"m1"->"d2"
"m1"->"d3"
"m1"->"d4"
"m2"->"d1"
"m2"->"d2"
"m2"->"d3"
"m2"->"d4"
"d1"->"r0"
"d1"->"r1"
"d1"->"r2"
"d1"->"r3"
"d1"->"r4"
"d1"->"r5"
"d1"->"r6"
"d1"->"r7"
"d1"->"r8"
"d1"->"r9"
"d2"->"r0"
"d2"->"r1"
"d2"->"r2"
"d2"->"r3"
"d2"->"r4"
"d2"->"r5"
"d2"->"r6"
"d2"->"r7"
"d2"->"r8"
"d2"->"r9"
"d3"->"r0"
"d3"->"r1"
"d3"->"r2"
"d3"->"r3"
"d3"->"r4"
"d3"->"r5"
"d3"->"r6"
"d3"->"r7"
"d3"->"r8"
"d3"->"r9"

"d4"->"r0"
"d4"->"r1"
"d4"->"r2"
"d4"->"r3"
"d4"->"r4"
"d4"->"r5"
"d4"->"r6"
"d4"->"r7"
"d4"->"r8"
"d4"->"r9"
}

```
<!-- slide vertical=true-->
Why：
<h3 style="color:red">保持输入和输出一致！！！</h3>

<!-- slide vertical=true-->
<h1 style="color:red">脑子秀逗了???</h3>

<!-- slide vertical=true-->
### 文青的解释


<!-- slide vertical=true-->
声律启蒙
梅酸对李苦，青眼对白眉
```dot
digraph dui {
rankdir=LR;
label = "文青的神经网络" ;
"曹操说梅酸"->"梅酸"
"王戎说李苦"->"李苦"
"阮籍青眼"->"青眼"
"马良白眉"->"白眉"
"梅酸"->"梅酸对李苦"
"李苦"->"梅酸对李苦"
"青眼"->"青眼对白眉"
"白眉"->"青眼对白眉"
"青眼对白眉"->"青 眼"
"青眼对白眉"->"白 眉"
"梅酸对李苦"->"梅 酸"
"梅酸对李苦"->"李 苦"
"青 眼"->"阮籍 青眼"
"白 眉"->"马良 白眉"
"梅 酸"->"曹操 说梅酸"
"李 苦"->"王戎 说李苦"
}
```

<!-- slide vertical=true-->
梅酸对李苦，青眼对白眉是能够复原的高度精简过的信息
同样，m1、m2 代表了全部的输入信息！！！
也就是说自动缩减了特征的维度~
带来了玩法的改变！！！

<!-- slide -->
## CNN
卷积神经网络
![CNN](assets/CNNArchitecture.jpg)
<!-- slide vertical=true-->
卷积：手电筒一块一块过
![手电筒](assets/cnn.jpg)
每次看到手电筒照到的那  **一块地方**

<!-- slide vertical=true-->
池化：近视眼心更宽
![近视眼](assets/pool.jpg)
n * n -> 1 * 1

<!-- slide vertical=true-->
### CNN应用
![手写识别](assets/MINST.png)
![猫狗分类](assets/catdog.jpg)


<!-- slide -->
## RNN
循环神经网络
![RNN](assets/RNN-unrolled.png)
原理：状态记忆
<!-- slide vertical=true-->
### LTSM
![LTSM](assets/LSTM3-chain.png)
原理：三重门
<!-- slide vertical=true-->
### RNN应用
#### 机器翻译
![机器翻译](assets/rnn1.png)

<!-- slide vertical=true-->
####  语音识别
![语音识别](assets/rnn2.jpg)

<!-- slide -->
## GAN
![GAN](assets/gan.png)
<!-- slide vertical=true-->
### GAN应用
#### DCGAN生成女朋友
![DCGAN生成女朋友](assets/girl.jpg)

<!-- slide -->
## 学习顺序
- 入门：简单易懂
- 经典：全面严谨
- Blog
- Github
- 论文
- 比赛
 ```dot
 digraph graph1 {
     学习->思考->行动->学习
}
 ```
<!-- slide vertical=true-->
### 视频
- [Tensorflow教程 by 莫烦](https://www.bilibili.com/video/av16001891/)
- [网易云课堂的深度学习微专业 by 吴恩达](http://mooc.study.163.com/smartSpec/detail/1001319001.htm)

<!-- slide vertical=true-->
### 书籍
#### 实战类
没错，随便买，反正你会去Github上下代码的~~~
#### 专业类
- [《白话深度学习与Tensorflow》](https://item.jd.com/12228460.html) by 高扬、卫峥
- [《深度学习》](https://item.jd.com/12128543.html) by Ian Goodfellow、Yoshua Bengio 和 Aaron Courville
[电子版](https://github.com/exacity/deeplearningbook-chinese/releases/download/v0.5-beta/dlbook_cn_v0.5-beta.pdf)

#### 科普类
- [《终极算法》](https://item.jd.com/12079958.html) by Pedro Domingos


<!-- slide data-background-image="https://i.loli.net/2017/07/12/5965b7edd3a2a.jpeg" -->
 - <p style="color: #fff;">感谢Ian Goodfellow、Yoshua Bengio 和 Aaron Courville三巨头！</p> <!-- .element: class="fragment" data-fragment-index="3" -->
 - <p style="color: #fff;">感谢吴恩达怪大叔！</p> <!-- .element: class="fragment" data-fragment-index="2" -->
 - <p style="color: #fff;">感谢吴沫凡小哥哥！</p> <!-- .element: class="fragment" data-fragment-index="1" -->

<!-- slide data-background-image="assets/thankyou.jpg" data-transition="zoom" -->

<!-- slide vertical=true data-background-image="深度学习入门.png"-->
