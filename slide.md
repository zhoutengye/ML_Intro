title: 机器学习简介
speaker: Zhouteng Ye
js:
    - https://www.echartsjs.com/asset/theme/shine.js
css:
    - ./css/bilibili.css
plugins:
    - echarts
    - katex
    - mermaid

<slide class="bg-black-blue aligncenter" image="img/zju.jpeg .dark">

# 机器学习在海岸工程中的应用 {.text-landing.text-shadow}

## （一）机器学习简介 {.text-landing.text-shadow}

By 叶洲腾 {.text-intro}

演示文档地址：158.247.219.109

[:fa-github: Github](https://github.com/zhoutengye/ML_Intro){.button.ghost}

<slide class="bg-black-blue aligncenter" image="img/zju.jpeg .dark">

<video width="100%" height="100%" controls>
    <source src="hms/gandolf.mp4" type="video/mp4">
</video>

<slide class="bg-black-blue" :class="size-40 alignleft" image="img/zju.jpeg .dark">

# 机器学习简介 {.text-landing.text-shadow}

### + 什么是人工智能？

* 机器学习历史？ {.animated.fadeInUp.delay-400}
* 什么是人工智能？ {.animated.fadeInUp.delay-800}
* 机器学习?深度学习？神经网络？大数据？ {.animated.fadeInUp.delay-1200}
* 面对即将可能到来的人工智能时代，我们应当做什么？ {.animated.fadeInUp.delay-1600}

### + 机器学习基础

* 机器学习的过程 {.animated.fadeInUp.delay-400}
* 机器学习的关键要素 {.animated.fadeInUp.delay-800}
* 线性模型 {.animated.fadeInUp.delay-1200}
* 前馈神经网络模型 {.animated.fadeInUp.delay-1600}
* 卷积神经网络模型 {.animated.fadeInUp.delay-1600}

演示文档地址：158.247.219.109

<slide class="bg-black-blue aligncenter">
## 真“人工”智能？

<div class="aspect-ratio">
<iframe src="https://www.youtube.com/embed/FGQapTlrsR4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
来源：Youtube MsDugie频道，绝命毒师S4E13

<slide class="bg-black-blue aligncenter">

## 人工智能 v1.0
因特尔公司为霍金打字/说话设计的黑科技
<div class="aspect-ratio" align="center">
<iframe  src="https://www.youtube.com/embed/OTmPw4iy0hk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

来源：Youtube Bloomberd Quicktake频道

<slide class="bg-black-blue aligncenter">

## 人工智能 v2.0?
Nueralink 4月9日公布脑机接口成果

<div class="aspect-ratio">
    <iframe src="https://player.bilibili.com/player.html?aid=972515657&bvid=BV16p4y1879K&cid=323028604&page=1&high_quaility=1" scrolling="no" border="0" frameborder="yes" framespacing="0" allowfullscreen="true"> </iframe>
</div>

来源：Youtube Neuralink频道，Bilibili up主火星小石头翻译。




<slide class="aligncenter">

## 人工智能发展史

<embed src="./img/AI-history.svg" type="image/svg+xml" />

<font size=6>
  | 年份        | 事件                     | 备注                                                 |
  | :---------- | :----------------------- | :--------------------------------------------------- |
  | 1950        | 图灵测试                 | 发表论文《机器能思考吗？》                           |
  | 1956        | 达特茅斯会议             | 由John MaCarthy组织，标志着人工智能学科诞生          |
  | 1950s-1970s | 第一次繁荣               | 符号推理主导，从数理逻辑到专家系统                   |
  | 1970s 后期  | 第一次低谷               | 对于符号系统盲目乐观，最终复杂度过高，远无法达到预期 |
  | 1980s-1990s | 第二次繁荣               | 在符号推理的基础上专家系统（推理+知识）              |
  | 1990s 初期  | 第二次衰落               | 专家系统逐渐达到极限                                 |
  | 1997年      | 深蓝战胜国际象棋世界冠军 | 人工智能开始复苏                                     |
  | 2006年      | 开始第三次繁荣           | Hinton 在 Science 发表论文开启了深度学习进入工业界   |
  | 2014年      | ImageNet图像识别战胜人类 | 人工智能首次在图像识别领域战胜人类                   |
  | 2016年      | Alpha Go战胜围棋世界冠军 | 以深度学习为代表的人工智能开始广泛进入大众视野       |
</font>

<slide class="aligncenter">

## 什么是人工智能

<embed src="./img/AI-def.svg" type="image/svg+xml" />

<slide class="aligncenter">

## 什么是机器学习

<embed src="./img/ML-def.svg" type="image/svg+xml" />

<slide class="aligncenter">

## 机器学习算法及深度学习

<embed src="./img/ML-alg.svg" type="image/svg+xml" />

<slide class="alignleft">

## 常见名词之间的关系

<embed src="./img/relation.svg" type="image/svg+xml" />

&nbsp;

## 做个类比

<font size=6>
|              |          |                                                                  |
| ------------ | -------- | ---------------------------------------------------------------- |
| 人工智能     | 最终状态 | 世界最强的料理，能发光的那种                                     |
| 机器学习     | 方法论   | 一个强大的后厨管理系统（目前还需要人管，但是需要的人越来越少了） |
| 人工神经网络 | 方法     | 不错的厨具                                                       |
| 深度学习     | 方法     | 哇！金色传说厨具！                                               |
| 数据         | 基础建设 | 做菜原料                                                         |
| 大数据       | 新基建   | 自家农场，肉菜管够，多多益善,专供金色传说厨具                    |
</font>

<slide :class="size-60">

# 此次人工智能浪潮背后的原因

&nbsp;

:::shadowbox

## 计算机硬件的发展
GPU的快速发展带了计算能力快速的提升; <br>
同时，计算机存储和互联网和高速发展为更多的数据交互和存储提供了条件。

---

## 大数据时代
互联网时代到移动互联网时代，产生了超从前的数据; <br>
数据的作用也越来越被得到重视，各领域开始加大对数据科学的投入。

---

## 以深度学习为代表的机器学习算法的发展
深度学习对于大数据的处理能力强于其他算法，在某些领域获得了惊人的表现。

---

## 大笔资金的持续投入
近些年，各国政府，企业都投入了大笔资金到了人工智能领域;
<br>
与前两次浪潮不同，以深度学习为代表的人工智能走出了学术界，在工业界得到了广泛应用，并且产生了巨大的收益。

<slide :class="size-60">

# 此次人工智能浪潮的特点

&nbsp;

:::shadowbox

## 发展迅速
此次人工智能的发展速度大家有目共睹，
尤其是近5年来，人工智能对生活的很多方面产生了巨大改变。
<br>
新的算法不断提出，应用不断扩展。

---

## 普及面广
移动互联网时代，人工智能的可以说是是无处不在。
<br>
不仅体现于受到受众的普及面广，使用者的数量也在急剧增加。

---

## 表现出众
虽然深度学习的结果往往难以解释，
且十分依赖于经验性的参数调整;
<br>
实践结果表明，对于特定问题，
深度学习经常有着碾压其他方法的表现。

---

## 门槛低
同时由于各大互联网公司的努力，
机器学习相关软件包的使用变得越来越容易。
<br>
数据科学的从业者，往往不需要经过很高门槛的培训，
就可以在实际工作中获得很好的表现。

<slide :class="size-60">

# 数据霸权？
<br>
#### “事实上，我们最后可能会得到连奥威尔都难以想象的结果：完全的监控政权，不但追踪所有人的一切行为和话语，甚至还能进入我们体内，掌握我们内心的感受。举例来说，如果某国领导人拿到这项新技术，想想看他会怎么运用？在未来，可能该国所有公民都会被要求佩戴生物统计手环，不仅监控他们的一言一行，还掌握他们的血压和大脑活动。而且，随着科学越来越了解大脑，并运用机器学习的庞大力量，该国政权可能会有史以来第一次真正知道每个公民在每个时刻想些什么。如果你看到一张该国领导人的照片，而生物传感器发现你出现愤怒的迹象（血压升高、杏仁核活动增加），大概明早你就会被抓进监狱。”

<br>
#### -- 尤瓦尔·赫拉利. 《今日简史：人类命运大议题》

<slide :class="size-60">

# 非计算机/数据科学从业者当下可以做什么

<br>

#### 虽然人工智能未来的发展仍然不是很明朗，可以确定的是，此次人工智能浪潮必然带来生产力的大幅度提升，越来越多的重复性的劳动会被替代，也会有新的职位形态产生。类似于汽车的发明取代了马车，虽然车夫可以改行做司机，你如何保证你的角色是车夫，而不是马呢？当下我们可以做的有：

<br>

### 1. 尝试了解一些数据科学相关基础知识

### 2. 掌握一些比较基本的机器学习技能

### 3. 根据自己实际的科研/工作需求，让机器学习提升自己的生产力

<slide class="bg-black-blue aligncenter">

### 一个很有意思的应用

<div class="aspect-ratio">
    <iframe src="//player.bilibili.com/player.html?aid=83410956&bvid=BV1FJ411J7Uu&cid=142696182&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</div>

<slide class="bg-black-blue" :class="size-40 alignleft" image="img/zju.jpeg .dark">

# 机器学习简介 {.text-landing.text-shadow}

* 机器学习的过程 {.animated.fadeInUp}
* 机器学习的关键要素 {.animated.fadeInUp.delay-400}
* 线性模型 {.animated.fadeInUp.delay-400}
* 前馈神经网络模型 {.animated.fadeInUp.delay-400}
* 卷积神经网络模型 {.animated.fadeInUp.delay-800}

<slide :class="size-60">

#### “神射手假说”：设有一名神枪手，在一个靶子上每隔十厘米打一个洞。设想这个靶子的平面上生活着一种二维智能生物，它们中的科学家在对自己的宇宙进行观察后，发现了一个伟大的定律：“宇宙每隔十厘米，必然会有一个洞。”它们把这个神枪手一时兴起的随意行为，看成了自己宇宙中的铁律。

<br>

#### “农场主假说”：一个农场里有一群火鸡，农场主每天中午十一点来给它们喂食。火鸡中的一名科学家观察这个现象，一直观察了近一年都没有例外，于是它也发现了自己宇宙中的伟大定律：“每天上午十一点，就有食物降临。”它在感恩节早晨向火鸡们公布了这个定律，但这天上午十一点食物没有降临，农场主进来把它们都捉去杀了。

<br>
#### -- 刘慈欣. 《三体》

<slide :class="size-60 aligncenter">

# 问题

<br>

## 假如要让机器学习得到上述假说，该怎么做呢？

<slide :class="size-60">

## 机器学习农场主假说

<embed src="./img/farmer.svg" type="image/svg+xml" />


<slide :class="size-60">

## 机器学习神射手假说

<embed src="./img/shooter.svg" type="image/svg+xml" />

<slide :class="size-60">

# 有监督学习的过程

<embed src="./img/ML-procedure.svg" type="image/svg+xml" />

### 简单不严谨来说就是：{.tobuild.MoveIn}


### - 接：描述数学问题，处理训练数据 {.tobuild.MoveIn}

### - 化：对数据进行训练，得到目标函数 $\vec y = f(\vec x)$  {.tobuild.MoveIn}

### - 发：根据训练确定的函数 $\vec y = f(\vec x)$ 进行预测  {.tobuild.MoveIn}

<slide :class="size-60">

# 机器学习和统计模型方法对比

<br>

## 相同点

#### - 均基于数据分析
#### - 均以概率为基础

<br>

## 区别

#### - 机器学习基于部分统计学，但是不只有统计学

#### - 机器学习更关心预测能力，统计模型更关心变量之间的关系

#### - 传统统计模型方法没有训练集和预测集概念，机器学习有

#### - 统计模型往往有简介的显式表达式，机器学习的模型往往没有

#### - 从计算和预测角度来看，统计模型能做的机器学习一定可以做，机器学习可以做的统计模型未必可以做（尤其是高维度问题）


<slide :class="size-60">

# 如何定义机器学习模型的问题

#### 对于有监督学习，机器学习要解决的问题往往可以用 $\vec y=f(\vec x)$ 来描述。那么如何定义如下问题？

#### 有如下两句话：

#### - 这杯水是半满的
#### - 这杯水是半空的

#### 假如要预测的 $\vec y$ 是杯子里有多少水，似乎相对容易。

#### 但是如果要预测的是这两句话对于阅读者的感受呢？

<slide :class="size-60">

# 目标问题的维度如何定义

:::column {.sm}

![](/img/answer.png) {.tobuild.MoveIn}

---

## 二维码的维度是多少？ {.tobuild.MoveIn}

#### 对于 $25 \times 25$ 像素的二维码，其像素维度为 625。 {.tobuild.MoveIn}

## 二维码能用完吗？ {.tobuild.MoveIn}

#### 每个像素只有黑和白两种状态，625个像素能表示的所有排列组合是 $2^{625}$ 个。根据腾讯数据，微信一年消耗6000亿个二维码，估算全球数据，大胆地给乘以1万倍吧，那么人类以当前速度要消耗掉 $25 \times 25$ 所有二维码需要多少年呢？{.tobuild.MoveIn}

#### 答案是需要 $n= \frac{2^{625}}{6\times 10^{13}}=1.39 \times 10^{175}$  年{.tobuild.MoveIn}

#### 那么二维码能用完吗？扫一扫二维码就知道。{.tobuild.MoveIn}
{.tobuild.MoveIn}

---

:::

# 维数灾难 {.tobuild.MoveIn}

#### 定义：当（数学）空间维度增加时，分析和组织高维空间（通常有成百上千维），因体积指数增加而遇到各种问题场景。{.tobuild.MoveIn}


<slide :class="size-60">

# 如何定义距离

### 最常见关于距离的定义是欧几里徳距离。 除此之外，还有很多其他的距离定义。例如：

### - 测地线距离
### - 曼哈顿距离
### - 人之间的度分距离

### 那么如何在数据层面上定义的距离呢？或者说，如何判定机器学习训练得到的函数结果和数据集最接近？

### 以下问题中距离该如何定义？

### - 两张图像之间的距离
### - 两句话之间的距离

<slide :class="size-60">

# 数据预处理

## - 数据清理

## - 数据集成

## - 数据增强

## - 数据规约

## - 数据变换
#### * 归一化
#### * 标准化

<slide :class="size-60">

# 过拟合

#### 过拟合指的是紧密或精确地匹配特定数据集，以致于无法良好地拟合其他数据或预测未来的观察结果的现象。在机器学习中表现为训练数据集中表现优秀，测试，预测数据集中表现不佳。

### 防止过拟合的办法：

#### - 交叉验证
#### - 特定方法各自的处理办法，例如决策树的剪枝，卷积神经网络的断连等

<slide :class="size-60">

# 单变量线性回归模型 

![](/img/linear.png)
图片来自： Machine Learning 课件 by Andrew Ng

$$y=ax+b$$

<slide :class="size-60">

# 两变量逻辑回归分类

:::column {.ms}

图片来自： Machine Learning 课件 by Andrew Ng

![](/img/logistic.png)

---

![](/img/sigmoid.png)

### 其中，
$$g(z) = \frac{1}{1+e^{-z}}$$

:::


<slide :class="size-60">


# 人工神经网络

### 感知器/神经元

![](/img/neural.png)
图片来自： Machine Learning 课件 by Andrew Ng

<slide :class="size-60">

### 人工神经网络

![](/img/NeuralNetwork.png)
图片来自： Machine Learning 课件 by Andrew Ng

<slide :class="size-60">

# 基于sklearn的手写识别程序

## [html 版本](hms/MINST.html)

## [Jupyter 版本（8888端口）](localhost:8888)

<slide :class="size-60">

# 神经网络在图像识别领域的局限

![](/img/eight1.gif)
图片来自 Adam Geitgey的博客

![](/img/eight2.png)



<slide :class="alignleft">

# 神经网络在图像识别领域的局限

### 图片的像素将由二维变为一维排列进入到神经网络的输入层。MINST数据集的图片均为大小接近且居中的手写图片，对于手写位置偏移，或者大小不一致的情况，即使是完全一样的数字，以一维向量的视角看是完全不同。

:::column

![](/img/eight4.gif)
图片来自 Adam Geitgey的博客

## 解决方法一：滑框法

### 缺陷：
#### - 仅在某些特定情况下有效

----

![](/img/eight5.png)

## 解决方法二：增加训练数据

### 缺陷：
#### - 数据量和网络规模都大幅度增加

:::

<slide :class="size-60">

# 卷积神经网络

### 卷积神经网络雏形

![](/img/cnn.png)

图片来自https://zhuanlan.zhihu.com/p/22094600

<slide :class="size-60">

# 卷积神经网络

### Yan LeCun 1998提出最早的卷积神经网络 
![](/img/LeNet.png)

###  AlexNet
![](/img/AlexNet.png)

图片来自https://zhuanlan.zhihu.com/p/22094600

<slide :class="size-60">

# 卷积神经网络

### GoogleNet
![](/img/GoogleNet.png)

### ResNet
![](/img/ResNet.png)

图片来自https://zhuanlan.zhihu.com/p/22094600

<slide :class="size-100">

# 卷积神经网络关键操作

:::column {.ms}

## 卷积

![](/img/convolution.gif)

图片来自 http://cs231n.github.io/convolutional-networks/

---

## 池化

![](/img/pooling.jpeg)

图片来自 https://cs231n.github.io/convolutional-networks/

:::

<slide :class="size-60">

# 关于卷积和池化的简单交互演示

## [html 版本](hms/ConvPool.html)

## [Jupyter 版本（8888端口）](localhost:8888)


<slide :class="size-60">

# 可视化卷积神经网络网站

<br>

## 由佐治亚理工大学 Zijie Jay Wang 开发

## [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

<slide :class="size-60">

# 总结

### - 没有最好的机器学习算法，只有最合适的机器学习算法
### - 对目标问题本身和数据科学的理解和经验对决定机器学习的表现有着重大影响
### - 正确地使用机器学习可以大幅度提升解决问题的能力
### - 机器学习上手十分容易
### - 大多数人不需要改进机器学习算法，只是拿来使用
### - 也许不久后的将来，使用机器学习就和开车一样，成为很多人学习和工作中的必备技能


<slide class="bg-black-blue aligncenter" image="img/gandolf.jpg .dark">

# 感谢观看 {.text-landing.text-shadow}
