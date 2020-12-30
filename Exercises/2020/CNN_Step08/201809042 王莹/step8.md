>>>>>>>> ## step8卷积神经网络 知识总结及实验代码分析
* 本章节将会逐步介绍卷积的前向计算、卷积的反向传播、池化的前向计与反向传播，然后用代码实现一个卷积网络并训练一些实际数据。
* 侧重点：侧重于讲解神经网络的工作原理

>## 0.卷积神经网络概述
#### 卷积神经网络的能力
卷积神经网络（CNN，Convolutional Neural Net)是神经网络的类型之一，在图像识别和分类领域中取得了非常好的效果，比如识别人脸、物体、交通标识等，这就为机器人、自动驾驶等应用提供了坚实的技术基础。
* 以下是卷积神经网络在日常生活中识别各种物体的实例图片。
  <img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/boat_people.png">
  ###### （根据本长图片可以看到通过应用卷积神经网络，它识别出了图片中有一条船以及船上有四个人）

  ##### 识别物体和给出简要的场景描述是两套系统配合才能完成的任务，第一个系统只负责识别，第二个系统可以根据第一个系统的输出形成摘要文字。

  #### 卷积神经网络的典型结构
* 如下图是一个典型的卷积神经网络结构图
  <img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_net.png" />
  
  分析结构图的图层结构
 1. 原始的输入是一张图片，可以是彩色的，也可以是灰度的或黑白的。这里假设是只有一个通道的图片，目的是识别0~9的手写体数字；
2. 第一层卷积，我们使用了4个卷积核，得到了4张feature map；激活函数层没有单独画出来，这里我们紧接着卷积操作使用了Relu激活函数；
3. 第二层是池化，使用了Max Pooling方式，把图片的高宽各缩小一倍，但仍然是4个feature map；
4. 第三层卷积，我们使用了4x6个卷积核，其中4对应着输入通道，6对应着输出通道，从而得到了6张feature map，当然也使用了Relu激活函数；
5. 第四层再次做一次池化，现在得到的图片尺寸只是原始尺寸的四分之一左右；
6. 第五层把第四层的6个图片展平成一维，成为一个fully connected层；
7. 第六层再接一个小一些的fully connected层；
8. 最后接一个softmax函数，判别10个分类。

所以，在一个典型的卷积神经网络中，会至少包含以下几个层：

- 卷积层
- 激活函数层
- 池化层
- 全连接分类层
  
#### 卷积核的作用
* 卷积核。卷积网络之所以能工作，完全是卷积核的功劳。
* 卷积核的具体作用
  <img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/circle_filters.png" ch="500" />
  图中所示的内容，是使用9个不同的卷积核在同一张图上运算后得到的结果，而表中按顺序列出了9个卷积核的数值和名称，可以一一对应到上面的9张图中。
  卷积的效果

||1|2|3|
|---|---|---|---|
|1|0,-1, 0<br>-1, 5,-1<br>0,-1, 0|0, 0, 0 <br> -1, 2,-1 <br> 0, 0, 0|1, 1, 1 <br> 1,-9, 1 <br> 1, 1, 1|
||sharpness|vertical edge|surround|
|2|-1,-2, -1 <br> 0, 0, 0<br>1, 2, 1|0, 0, 0 <br> 0, 1, 0 <br> 0, 0, 0|0,-1, 0 <br> 0, 2, 0 <br> 0,-1, 0|
||sobel y|nothing|horizontal edge|
|3|0.11,0.11,0.11 <br>0.11,0.11,0.11<br>0.11,0.11,0.11|-1, 0, 1 <br> -2, 0, 2 <br> -1, 0, 1|2, 0, 0 <br> 0,-1, 0 <br> 0, 0,-1|
||blur|sobel x|embossing|
* 中间的图叫做"nothing"，它与原图一样可以作为对比参考使用。
* 各个卷积核的作用

|序号|名称|说明|
|---|---|---|
|1|锐化|如果一个像素点比周围像素点亮，则此算子会令其更亮|
|2|检测竖边|检测出了十字线中的竖线，由于是左侧和右侧分别检查一次，所以得到两条颜色不一样的竖线|
|3|周边|把周边增强，把同色的区域变弱，形成大色块|
|4|Sobel-Y|纵向亮度差分可以检测出横边，与横边检测不同的是，它可以使得两条横线具有相同的颜色，具有分割线的效果|
|5|Identity|中心为1四周为0的过滤器，卷积后与原图相同|
|6|横边检测|检测出了十字线中的横线，由于是上侧和下侧分别检查一次，所以得到两条颜色不一样的横线|
|7|模糊|通过把周围的点做平均值计算而“杀富济贫”造成模糊效果|
|8|Sobel-X|横向亮度差分可以检测出竖边，与竖边检测不同的是，它可以使得两条竖线具有相同的颜色，具有分割线的效果|
|9|浮雕|形成大理石浮雕般的效果|

#### 卷积后续的运算
* 卷积神经网络通过反向传播而令卷积核自我学习，找到分布在图片中的不同的feature，最后形成的卷积核中的数据。但是如果想达到这种效果，只有卷积层的话是不够的，还需要激活函数、池化等操作的配合。
  <img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/circle_conv_relu_pool.png" ch="500" />
###### 这张图片依次展示了：

1. 原图
2. 卷积结果
3. 激活结果
4. 池化结果
注意图一是原始图片，用cv2读取出来的图片，其顺序是反向的，即：

- 第一维是高度
- 第二维是宽度
- 第三维是彩色通道数，但是其顺序为BGR，而不是常用的RGB

1. 我们对原始图片使用了一个3x1x3x3的卷积核，因为原始图片为彩色图片，所以第一个维度是3，对应RGB三个彩色通道；我们希望只输出一张feature map，以便于说明，所以第二维是1；我们使用了3x3的卷积核，用的是sobel x算子。所以图二是卷积后的结果。

2. 图三做了一层Relu激活计算，把小于0的值都去掉了，只留下了一些边的特征。

3. 图四是图三的四分之一大小，虽然图片缩小了，但是特征都没有丢失，反而因为图像尺寸变小而变得密集，亮点的密度要比图三大而粗。
   
#### 卷积神经网络的学习
* 全连接层是做为分类层使用。
* 在最后一层的池化后面，把所有特征数据变成一个一维的全连接层，然后就和普通的深度全连接网络一样了，通过在最后一层的softmax分类函数，以及多分类交叉熵函数，对比图片的OneHot编码标签，回传误差值，从全连接层传回到池化层，通过激活函数层再回传给卷积层，对卷积核的数值进行梯度更新，实现卷积核数值的自我学习。
* 同一个背景下数字8的大小、位置、形状的不同 对于人类的视觉系统来说都可以轻松应对，即平移不变性、旋转视角不变性、尺度不变性。
* 那么卷积神经网络网络的处理方法如下：
  <img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/8.png" ch="500" />


- 平移不变性
  
  对于原始图A，平移后得到图B，对于同一个卷积核来说，都会得到相同的特征，这就是卷积核的权值共享。但是特征处于不同的位置，由于距离差距较大，即使经过多层池化后，也不能处于近似的位置。此时，后续的全连接层会通过权重值的调整，把这两个相同的特征看作同一类的分类标准之一。如果是小距离的平移，通过池化层就可以处理了。

- 旋转不变性

  对于原始图A，有小角度的旋转得到C，卷积层在A图上得到特征a，在C图上得到特征c，可以想象a与c的位置间的距离不是很远，在经过两层池化以后，基本可以重合。所以卷积网络对于小角度旋转是可以容忍的，但是对于较大的旋转，需要使用数据增强来增加训练样本。一个极端的例子是当6旋转90度时，谁也不能确定它到底是6还是9。

- 尺度不变性

  对于原始图A和缩小的图D，人类可以毫不费力地辨别出它们是同一个东西。池化在这里是不是有帮助呢？没有！因为神经网络对A做池化的同时，也会用相同的方法对D做池化，这样池化的次数一致，最终D还是比A小。如果我们有多个卷积视野，相当于从两米远的地方看图A，从一米远的地方看图D，那么A和D就可以很相近似了。这就是Inception的想法，用不同尺寸的卷积核去同时寻找同一张图片上的特征。

#### 代码及运行结果
##### ch17,level10(使用9个不同的卷积核在同一张图上运算)
* 代码如下
  [![D5FE3q.png](https://s3.ax1x.com/2020/12/02/D5FE3q.png)](https://imgchr.com/i/D5FE3q)
  
* 实验结果图
 [![D5pwLQ.png](https://s3.ax1x.com/2020/12/02/D5pwLQ.png)](https://imgchr.com/i/D5pwLQ)

> ## 1. 卷积的前向计算
#### 卷积的数学定义
##### 连续定义

$$h(x)=(f*g)(x) = \int_{-\infty}^{\infty} f(t)g(x-t)dt \tag{1}$$
##### f(x)∗g(x) 表示 f(x) 和 g(x) 的卷积，注意此处自变量为 x；它是对 (−∞,∞) 区间上对 τ 求积分；积分对象为两个函数的乘积：f(τ) 和 g(x−τ)。等式右边只有 g(x−τ) 提到了 x，其他部分都在关注 τ


卷积与傅里叶变换有着密切的关系。利用这点性质，即两函数的傅里叶变换的乘积等于它们卷积后的傅里叶变换，能使傅里叶分析中许多问题的处理得到简化。

##### 离散定义

$$h(x) = (f*g)(x) = \sum^{\infty}_{t=-\infty} f(t)g(x-t) \tag{2}$$
#### 一维卷积实例
* 书本中举证了两个骰子点数相加为指定数字的概率计算问题。这里不重复赘述了。
最后经过计算再与卷积公式对比可以得出结论他们符合卷积的定义：
（骰子点数是离散的，所以最终卷积的离散公式）
$$h(x) = (f*g)(x) = \sum^{\infty}_{t=-\infty} f(t)g(x-t) \tag{2}$$
#### 单入单出的二维卷积
如果把图像Image简写为$I$，把卷积核Kernal简写为$K$，则目标图片的第$(i,j)$个像素的卷积值为：

$$
h(i,j) = (I*K)(i,j)=\sum_m \sum_n I(m,n)K(i-m,j-n) \tag{3}
$$

可以看出，这和一维情况下的公式2是一致的。从卷积的可交换性，我们可以把公式3等价地写作：

$$
h(i,j) = (I*K)(i,j)=\sum_m \sum_n I(i-m,j-n)K(m,n) \tag{4}
$$

公式4的成立，是因为我们将Kernal进行了翻转。在神经网络中，一般会实现一个互相关函数(corresponding function)，而卷积运算几乎一样，但不反转Kernal：

$$
h(i,j) = (I*K)(i,j)=\sum_m \sum_n I(i+m,j+n)K(m,n) \tag{5}
$$

在图像处理中，自相关函数和互相关函数定义如下：

- 自相关：设原函数是f(t)，则$h=f(t) \star f(-t)$，其中$\star$表示卷积
- 互相关：设两个函数分别是f(t)和g(t)，则$h=f(t) \star g(-t)$

互相关函数的运算，是两个序列滑动相乘，两个序列都不翻转。卷积运算也是滑动相乘，但是其中一个序列需要先翻转，再相乘。所以，从数学意义上说，机器学习实现的是互相关函数，而不是原始含义上的卷积。但我们为了简化，把公式5也称作为卷积。这就是卷积的来源。
#### 卷积的运算过程
一张图片，通常是彩色的，具有红绿蓝三个通道。我们可以有两个选择来处理：

1. 变成灰度的，每个像素只剩下一个值，就可以用二维卷积
2. 对于三个通道，每个通道都使用一个卷积核，分别处理红绿蓝三种颜色的信息（三维卷积，即有三个卷积核分别对应书的三个通道，三个子核的尺寸是一样的，比如都是2x2，这样的话，这三个卷积核就是一个3x2x2的立体核，称为过滤器Filter，所以称为三维卷积。）
虽然输入图片是多个通道的，或者说是三维的，但是在相同数量的过滤器的计算后，相加在一起的结果是一个通道，即2维数据，所以称为降维。这当然简化了对多通道数据的计算难度，但同时也会损失多通道数据自带的颜色信息。
#### 多入多出的同维卷积
Feature-m,n还用红绿蓝三色表示，是因为在此时，它们还保留着红绿蓝三种色彩的各自的信息，一旦相加后得到Result，这种信息就丢失了。
#### 卷积编程模型
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv3d.png" ch="500" />

###### 三通道经过两组过滤器的卷积过程
1. 每个Input Channel就是特征输入，在上图中是3个
2. 卷积层的卷积核相当于隐层的神经元，上图中隐层有2个神经元
3. $W(m,n), m=[1,2], n=[1,3]$相当于隐层的权重矩阵$w_{11},w_{12},......$
4. 每个卷积核（神经元）有1个偏移值
对于三维卷积，有以下特点：

1. 预先定义输出的feature map的数量，而不是根据前向计算自动计算出来，此例中为2，这样就会有两组WeightsBias
2. 对于每个输出，都有一个对应的过滤器Filter，此例中Feature Map-1对应Filter-1
3. 每个Filter内都有一个或多个卷积核Kernal，对应每个输入通道(Input Channel)，此例为3，对应输入的红绿蓝三个通道
4. 每个Filter只有一个Bias值，Filter-1对应b1，Filter-2对应b2
5. 卷积核Kernal的大小一般是奇数如：1x1, 3x3, 5x5, 7x7等，此例为5x5
#### 步长 stride
每次计算后，卷积核会向右或者向下移动一个单元，即步长stride = 1
#### 填充 padding
卷积后图片尺寸改变，想恢复原图大小，可使用填充。
一般我们会向原始图片周围填充一圈0，然后再做卷积。

>## 2 卷积前向计算代码实现
#### 卷积核的实现
卷积核，实际上和全连接层一样，是权重矩阵加偏移向量的组合，卷积核的权重矩阵是四维的。
* 如下为卷积核的组成
```Python
class ConvWeightsBias(WeightsBias_2_1):
    def __init__(self, output_c, input_c, filter_h, filter_w, init_method, optimizer_name, eta):
        self.FilterCount = output_c
        self.KernalCount = input_c
        self.KernalHeight = filter_h
        self.KernalWidth = filter_w
        ...

    def Initialize(self, folder, name, create_new):
        self.WBShape = (self.FilterCount, self.KernalCount, self.KernalHeight, self.KernalWidth)        
        ...
```
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/ConvWeightsBias.png" />

##### 各个维度的数值如下：

- FilterCount=2，第一维，过滤器数量，对应输出通道数。
- KernalCount=3，第二维，卷积核数量，对应输入通道数。两个Filter里面的Kernal数必须相同。
- KernalHeight=5，KernalWidth=5，卷积核的尺寸，第三维和第四维。同一组WeightsBias里的卷积核尺寸必须相同。

在初始化函数中，会根据四个参数定义`WBShape`，然后在`CreateNew`函数中，创建相应形状的`Weights`和`Bias`。
#### 卷积前向运算的实现
* 一共有三种方法，在step8中有具体的讲解和介绍
###### 在方法一中
1. 批量数据循环（第一维）：`bs in batch_size`，对每个样本进行计算；
2. 输出通道循环（第二维）：`oc in output_channel`。这里先把`bias`加上了，后加也可以；
3. 输入通道循环：`ic in input_channel`;
4. 输出图像纵坐标循环：`i in out h`；
5. 输出图像横坐标循环：`j in out_w`。循环4和5完成对输出图像的每个点的遍历，在下面的子循环中计算并填充值；
6. 卷积核纵向循环（第三维）：`fh in filter_height`；
7. 卷积核横向循环（第四维）：`fw in filter_width`。循环6和7完成卷积核与输入图像的卷积计算，并保存到循环4和5指定的输出图像的点上。

我们试着运行上面的代码并循环10次，看看它的执行效率如何：

```
Time used for Python: 38.057225465774536
```
足足等了30多秒后，才返回结果。

通过试验发现，其运行速度非常慢，如果这样的函数在神经网络训练中被调用几十万次，其性能是非常糟糕的，这也是`Python`做为动态语言的一个缺点。
###### 在方法二中（方法一的基础上把它编译成静态方法）
* 这次只用了0.07秒，比纯`Python`代码快了500多倍！
* 为什么不把所有的Python代码都编译成C代码呢？
是因为`numba`的能力有限，并不支持`numpy`的所有函数，所以只能把关键的运算代码设计为独立的函数，然后用`numba`编译执行，函数的输入可以是数组、向量、标量，不能是复杂的自定义结构体或函数指针。

###### 在方法三中（把卷积操作转换为矩阵操作）
* 在Caffe框架中，巧妙地把逐点相乘的运算转换成了矩阵运算，大大提升了程序运行速度。这就是著名的`im2col`函数

#### 代码及运行结果
* Level2_Numba_Test.py测试`numba`库的性能
  [![DoANh4.png](https://s3.ax1x.com/2020/12/02/DoANh4.png)](https://imgchr.com/i/DoANh4)
  [![DoAeAS.png](https://s3.ax1x.com/2020/12/02/DoAeAS.png)](https://imgchr.com/i/DoAeAS)

* `Level2_Img2Col_Test`比较`numba`方法和`im2col`方法的前向计算性能
 [![DoV4Tf.png](https://s3.ax1x.com/2020/12/02/DoV4Tf.png)](https://imgchr.com/i/DoV4Tf)
 [![DoVHpQ.png](https://s3.ax1x.com/2020/12/02/DoVHpQ.png)](https://imgchr.com/i/DoVHpQ)



  在Level2的主过程中有4个函数：

- `test_2d_conv`，理解2维下`im2col`的工作原理
- `understand_4d_im2col`，理解4维下`im2col`的工作原理
- `test_4d_im2col`，比较两种方法的结果，从而验证正确性
- `test_performance`，比较两种方法的性能
> >## 3 卷积的反向传播原理
> #### 卷积层的训练
同全连接层一样，卷积层的训练也需要从上一层回传的误差矩阵，然后计算：

1. 本层的权重矩阵的误差项
2. 本层的需要回传到下一层的误差矩阵
  #### 计算反向传播的梯度矩阵
* 正向公式：
$$Z = W*A+b \tag{0}$$

其中，W是卷积核，*表示卷积（互相关）计算，A为当前层的输入项，b是偏移（未在图中画出），Z为当前层的输出项，但尚未经过激活函数处理。
有多个卷积核时的梯度计算

* 有多个卷积核也就意味着有多个输出通道。
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_2w2.png" ch="500" />
升维卷积

正向公式：

$$z_{111} = w_{111} \cdot a_{11} + w_{112} \cdot a_{12} + w_{121} \cdot a_{21} + w_{122} \cdot a_{22}$$
$$z_{112} = w_{111} \cdot a_{12} + w_{112} \cdot a_{13} + w_{121} \cdot a_{22} + w_{122} \cdot a_{23}$$
$$z_{121} = w_{111} \cdot a_{21} + w_{112} \cdot a_{22} + w_{121} \cdot a_{31} + w_{122} \cdot a_{32}$$
$$z_{122} = w_{111} \cdot a_{22} + w_{112} \cdot a_{23} + w_{121} \cdot a_{32} + w_{122} \cdot a_{33}$$

$$z_{211} = w_{211} \cdot a_{11} + w_{212} \cdot a_{12} + w_{221} \cdot a_{21} + w_{222} \cdot a_{22}$$
$$z_{212} = w_{211} \cdot a_{12} + w_{212} \cdot a_{13} + w_{221} \cdot a_{22} + w_{222} \cdot a_{23}$$
$$z_{221} = w_{211} \cdot a_{21} + w_{212} \cdot a_{22} + w_{221} \cdot a_{31} + w_{222} \cdot a_{32}$$
$$z_{222} = w_{211} \cdot a_{22} + w_{212} \cdot a_{23} + w_{221} \cdot a_{32} + w_{222} \cdot a_{33}$$

#### 有多个输入时的梯度计算
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_1W222.png" ch="500" />
 多个图层的卷积必须有一一对应的卷积核
所以有前向公式：

$$
\begin{aligned}
z_{11} &= w_{111} \cdot a_{111} + w_{112} \cdot a_{112} + w_{121} \cdot a_{121} + w_{122} \cdot a_{122}
\\\\
&+ w_{211} \cdot a_{211} + w_{212} \cdot a_{212} + w_{221} \cdot a_{221} + w_{222} \cdot a_{222} 
\end{aligned}
\tag{10}
$$
$$
\begin{aligned}
z_{12} &= w_{111} \cdot a_{112} + w_{112} \cdot a_{113} + w_{121} \cdot a_{122} + w_{122} \cdot a_{123} \\\\
&+ w_{211} \cdot a_{212} + w_{212} \cdot a_{213} + w_{221} \cdot a_{222} + w_{222} \cdot a_{223} 
\end{aligned}\tag{11}$$
$$
\begin{aligned}
z_{21} &= w_{111} \cdot a_{121} + w_{112} \cdot a_{122} + w_{121} \cdot a_{131} + w_{122} \cdot a_{132} \\\\
&+ w_{211} \cdot a_{221} + w_{212} \cdot a_{222} + w_{221} \cdot a_{231} + w_{222} \cdot a_{232} 
\end{aligned}\tag{12}$$
$$
\begin{aligned}
z_{22} &= w_{111} \cdot a_{122} + w_{112} \cdot a_{123} + w_{121} \cdot a_{132} + w_{122} \cdot a_{133} \\\\
&+ w_{211} \cdot a_{222} + w_{212} \cdot a_{223} + w_{221} \cdot a_{232} + w_{222} \cdot a_{233} 
\end{aligned}\tag{13}$$

最复杂的情况，求$J$对$a_{122}$的梯度：

$$
\begin{aligned}
\frac{\partial J}{\partial a_{111}}&=\frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial a_{122}} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial a_{122}} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial a_{122}} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial a_{122}}
\\\\
&=\delta_{z_{11}} \cdot w_{122} + \delta_{z_{12}} \cdot w_{121} + \delta_{z_{21}} \cdot w_{112} + \delta_{z_{22}} \cdot w_{111} 
\end{aligned}
$$

泛化以后得到：

$$\delta_{out1} = \delta_{in} * W_1^{rot180} \tag{14}$$

求$J$对$a_{222}$的梯度：

$$
\begin{aligned}
\frac{\partial J}{\partial a_{211}}&=\frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial a_{222}} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial a_{222}} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial a_{222}} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial a_{222}} \\\\
&=\delta_{z_{11}} \cdot w_{222} + \delta_{z_{12}} \cdot w_{221} + \delta_{z_{21}} \cdot w_{212} + \delta_{z_{22}} \cdot w_{211} 
\end{aligned}
$$

泛化以后得到：

$$\delta_{out2} = \delta_{in} * W_2^{rot180} \tag{15}$$
#### 偏移的梯度计算

根据前向计算公式1，2，3，4，可以得到：

$$
\begin{aligned}
\frac{\partial J}{\partial b} &= \frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial b} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial b} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial b} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial b}
\\\\
&=\delta_{z11} + \delta_{z12}  + \delta_{z21} + \delta_{z22} 
\end{aligned}
\tag{12}
$$

所以：

$$
\delta_b = \delta_{in} \tag{13}
$$

每个卷积核W可能会有多个filter，或者叫子核，但是一个卷积核只有一个偏移，无论有多少子核。
#### 代码及运行结果
[![DolrkR.png](https://s3.ax1x.com/2020/12/02/DolrkR.png)](https://imgchr.com/i/DolrkR)
[![DoltpV.png](https://s3.ax1x.com/2020/12/02/DoltpV.png)](https://imgchr.com/i/DoltpV)


>## 4.池化
* 常用池化方法
  池化方法分为两种，一种是最大值池化 Max Pooling，一种是平均值池化 Mean/Average Pooling。
  其目的是：

- 扩大视野：就如同先从近处看一张图片，然后离远一些再看同一张图片，有些细节就会被忽略
- 降维：在保留图片局部特征的前提下，使得图片更小，更易于计算
- 平移不变性，轻微扰动不会影响输出：比如上图中最大值池化的4，即使向右偏一个像素，其输出值仍为4
- 维持同尺寸图片，便于后端处理：假设输入的图片不是一样大小的，就需要用池化来转换成同尺寸图片

###### 一般我们都使用最大值池化。

##### Max Pooling
正向公式：

$$
w = max(a,b,e,f)
$$

反向公式（假设Input Layer中的最大值是b）：

$$
{\partial w \over \partial a} = 0, \quad {\partial w \over \partial b} = 1
$$

$$
{\partial w \over \partial e} = 0, \quad {\partial w \over \partial f} = 0
$$

因为a,e,f对w都没有贡献，所以偏导数为0，只有b有贡献，偏导数为1。

$$
\delta_a = {\partial J \over \partial a} = {\partial J \over \partial w} {\partial w \over \partial a} = 0
$$

$$
\delta_b = {\partial J \over \partial b} = {\partial J \over \partial w} {\partial w \over \partial b} = \delta_w \cdot 1 = \delta_w
$$

$$
\delta_e = {\partial J \over \partial e} = {\partial J \over \partial w} {\partial w \over \partial e} = 0
$$

$$
\delta_f = {\partial J \over \partial f} = {\partial J \over \partial w} {\partial w \over \partial f} = 0
$$

##### Mean Pooling

正向公式：

$$w = \frac{1}{4}(a+b+e+f)$$

反向公式（假设Layer-1中的最大值是b）：

$$
{\partial w \over \partial a} = \frac{1}{4}, \quad {\partial w \over \partial b} = \frac{1}{4}
$$

$$
{\partial w \over \partial e} = \frac{1}{4}, \quad {\partial w \over \partial f} = \frac{1}{4}
$$

因为a,b,e,f对w都有贡献，所以偏导数都为1：

$$
\delta_a = {\partial J \over \partial a} = {\partial J \over \partial w} {\partial w \over \partial a} = \frac{1}{4}\delta_w
$$

$$
\delta_b = {\partial J \over \partial b} = {\partial J \over \partial w} {\partial w \over \partial b} = \frac{1}{4}\delta_w
$$

$$
\delta_e = {\partial J \over \partial e} = {\partial J \over \partial w} {\partial w \over \partial e} = \frac{1}{4}\delta_w
$$

$$
\delta_f = {\partial J \over \partial f} = {\partial J \over \partial w} {\partial w \over \partial f} = \frac{1}{4}\delta_w
$$

无论是max pooling还是mean pooling，都没有要学习的参数，所以，在卷积网络的训练中，池化层需要做的只是把误差项向后传递，不需要计算任何梯度。

##### 代码及运行结果
[![Do8U6f.png](https://s3.ax1x.com/2020/12/02/Do8U6f.png)](https://imgchr.com/i/Do8U6f)
[![Do800g.png](https://s3.ax1x.com/2020/12/02/Do800g.png)](https://imgchr.com/i/Do800g)
>>>>>> ## 卷积神经网络应用
> ## 0. 经典的卷积神经网络模型
* 卷积神经网络从90年代的LeNet开始，沉寂了10年，也孵化了10年，直到2012年AlexNet开始再次崛起，后续的ZF Net、VGG、GoogLeNet、ResNet、DenseNet，网络越来越深，架构越来越复杂，解决反向传播时梯度消失的方法也越来越巧妙。
#####  LeNet (1998)
* LeNet是卷积神经网络的开创者LeCun在1998年提出，用于解决手写数字识别的视觉任务。自那时起，卷积神经网络的最基本的架构就定下来了：卷积层、池化层、全连接层。
##### AlexNet (2012)
* AlexNet$^{[4]}$网络结构在整体上类似于LeNet，都是先卷积然后在全连接。但在细节上有很大不同。AlexNet更为复杂。AlexNet有60 million个参数和65000个神经元，五层卷积，三层全连接网络，最终的输出层是1000通道的Softmax。
AlexNet的特点：

- 比LeNet深和宽的网络
  
  使用了5层卷积和3层全连接，一共8层。特征数在最宽处达到384。

- 数据增强
  
  针对原始图片256x256的数据，做了随机剪裁，得到224x224的图片若干张。

- 使用ReLU做激活函数
- 在全连接层使用DropOut
- 使用LRN
  
  LRN的全称为Local Response Normalizatio，局部响应归一化，是想对线性输出做一个归一化，避免上下越界。发展至今，这个技术已经很少使用了。
##### ZFNet (2013)

ZFNet$^{[5]}$是2013年ImageNet分类任务的冠军，其网络结构没什么改进，只是调了调参，性能较Alex提升了不少。ZF-Net只是将AlexNet第一层卷积核由11变成7，步长由4变为2，第3，4，5卷积层转变为384，384，256。
##### VGGNet (2015)

VGG Net$^{[6]}$由牛津大学的视觉几何组（Visual Geometry Group）和 Google DeepMind公司的研究员一起研发的的深度卷积神经网络，在 ILSVRC 2014 上取得了第二名的成绩，将 Top-5错误率降到7.3%。它主要的贡献是展示出网络的深度（depth）是算法优良性能的关键部分。目前使用比较多的网络结构主要有ResNet（152-1000层），GooleNet（22层），VGGNet（19层），大多数模型都是基于这几个模型上改进，采用新的优化算法，多模型融合等。到目前为止，VGG Net 依然经常被用来提取图像特征。
##### GoogLeNet (2014)

GoogLeNet$^{[7]}$在2014的ImageNet分类任务上击败了VGG-Nets夺得冠军，其实力肯定是非常深厚的，GoogLeNet跟AlexNet,VGG-Nets这种单纯依靠加深网络结构进而改进网络性能的思路不一样，它另辟幽径，在加深网络的同时（22层），也在网络结构上做了创新，引入Inception结构代替了单纯的卷积+激活的传统操作
####  ResNets (2015)

2015年何恺明推出的ResNet$^{[8]}$在ISLVRC和COCO上横扫所有选手，获得冠军。ResNet在网络结构上做了大创新，而不再是简单的堆积层数，ResNet在卷积神经网络的新思路，绝对是深度学习发展历程上里程碑式的事件。
#### ResNets (2015)

2015年何恺明推出的ResNet$^{[8]}$在ISLVRC和COCO上横扫所有选手，获得冠军。ResNet在网络结构上做了大创新，而不再是简单的堆积层数，ResNet在卷积神经网络的新思路，绝对是深度学习发展历程上里程碑式的事件。
> ## 1.实现颜色分类
> ##### 数据处理

由于输入图片是三通道的彩色图片，我们先把它转换成灰度图，

```Python
class GeometryDataReader(DataReader_2_0):
    def ConvertToGray(self, data):
        (N,C,H,W) = data.shape
        new_data = np.empty((N,H*W))
        if C == 3: # color
            for i in range(N):
                new_data[i] = np.dot(
                    [0.299,0.587,0.114], 
                    data[i].reshape(3,-1)).reshape(1,784)
        elif C == 1: # gray
            new_data[i] = data[i,0].reshape(1,784)
        #end if
        return new_data
```

向量[0.299,0.587,0.114]的作用是，把三通道的彩色图片的RGB值与此向量相乘，得到灰度图，三个因子相加等于1，这样如果原来是[255,255,255]的话，最后的灰度图的值还是255。如果是[255,255,0]的话，最后的结果是：

$$
\begin{aligned}
Y &= 0.299 \cdot R + 0.586 \cdot G + 0.114 \cdot B \\
&= 0.299 \cdot 255 + 0.586 \cdot 255 + 0.114 \cdot 0 \\
&=225.675
\end{aligned}
\tag{1}
$$

也就是说粉色的数值本来是(255,255,0)，变成了单一的值225.675。六种颜色中的每一种都会有不同的值，所以即使是在灰度图中，也会保留部分“彩色”信息，当然会丢失一些信息。这从公式1中很容易看出来，假设$B=0$，不同组合的$R、G$的值有可能得到相同的最终结果，因此会丢失彩色信息。

在转换成灰度图后，立刻用reshape(1,784)把它转变成矢量，该矢量就是每个样本的784维的特征值。

##### 搭建模型

我们搭建的前馈神经网络模型如下：

```Python
def dnn_model():
    num_output = 6
    max_epoch = 100
    batch_size = 16
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "color_dnn")
    
    f1 = FcLayer_2_0(784, 128, params)
    net.add_layer(f1, "f1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")

    f2 = FcLayer_2_0(f1.output_size, 64, params)
    net.add_layer(f2, "f2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    
    f3 = FcLayer_2_0(f2.output_size, num_output, params)
    net.add_layer(f3, "f3")
    s3 = ClassificationLayer(Softmax())
    net.add_layer(s3, "s3")

    return net
```

这就是一个普通的三层网络，两个隐层，神经元数量分别是128和64，一个输出层，最后接一个6分类Softmax。

##### 运行结果

训练100个epoch后，得到如下损失函数图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/color_dnn_loss.png" />

图18-15 训练过程中的损失函数值和准确度变化曲线

从损失函数曲线可以看到，此网络已经有些轻微的过拟合了，如果重复多次运行训练过程，会得到75%到85%之间的一个准确度值，并不是非常稳定，但偏差也不会太大，这与样本的噪音有很大关系，比如一条很细的红色直线，可能会给训练带来一些不确定因素。

最后我们考察一下该模型在测试集上的表现：

```
......
epoch=99, total_iteration=28199
loss_train=0.005832, accuracy_train=1.000000
loss_valid=0.593325, accuracy_valid=0.804000
save parameters
time used: 30.822062015533447
testing...
0.816
```
结果中出现了很多直线颜色被识别错误的问题
* 笔者推测的原因

1. 针对细直线，由于带颜色的像素点的数量非常少，被拆成向量后，这些像素点就会在1x784的矢量中彼此相距很远，特征不明显，很容易被判别成噪音；
2. 针对大色块，由于带颜色的像素点的数量非常多，即使被拆成向量，也会占据很大的部分，这样特征点与背景点的比例失衡，导致无法判断出到底哪个是特征点。
>#### 3.用卷积神经网络解决问题
#### 搭建模型

```Python
def cnn_model():
    num_output = 6
    max_epoch = 20
    batch_size = 16
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "color_conv")
    
    c1 = ConvLayer((3,28,28), (2,1,1), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    c2 = ConvLayer(p1.output_shape, (3,3,3), (1,0), params)
    net.add_layer(c2, "c2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2") 

    params.learning_rate = 0.1

    f3 = FcLayer_2_0(p2.output_size, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BnLayer(f3.output_size)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")
    
    f4 = FcLayer_2_0(f3.output_size, num_output, params)
    net.add_layer(f4, "f4")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net
```

表18-1展示了在这个模型中各层的作用和参数。

表18-1 模型各层的参数

|ID|类型|参数|输入尺寸|输出尺寸|
|---|---|---|---|---|
|1|卷积|2x1x1, S=1|3x28x28|2x28x28|
|2|激活|Relu|2x28x28|2x28x28|
|3|池化|2x2, S=2, Max|2x14x14|2x14x14|
|4|卷积|3x3x3, S=1|2x14x14|3x12x12|
|5|激活|Relu|3x12x12|3x12x12|
|6|池化|2x2, S=2, Max|3x12x12|3x6x6|
|7|全连接|32|108|32|
|8|归一化||32|32|
|9|激活|Relu|32|32|
|10|全连接|6|32|6|
|11|分类|Softmax|6|6|

为什么第一梯队的卷积用2个卷积核，而第二梯队的卷积核用3个呢？只是经过调参试验的结果，是最小的配置。如果使用更多的卷积核当然可以完成问题，但是如果使用更少的卷积核，网络能力就不够了，不能收敛。

#### 运行结果

经过20个epoch的训练后，得到的结果如图18-17。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/color_cnn_loss.png" />

图18-17 训练过程中的损失函数值和准确度变化曲线

以下是打印输出的最后几行：

```
......
epoch=19, total_iteration=5639
loss_train=0.005293, accuracy_train=1.000000
loss_valid=0.106723, accuracy_valid=0.968000
save parameters
time used: 17.295073986053467
testing...
0.963
```

可以看到我们在测试集上得到了96.3%的准确度，比前馈神经网络模型要高出很多，这也证明了卷积神经网络在图像识别上的能力
> #### 代码及运行结果
[![DoNUeK.png](https://s3.ax1x.com/2020/12/02/DoNUeK.png)](https://imgchr.com/i/DoNUeK)
 [![DoNJQ1.png](https://s3.ax1x.com/2020/12/02/DoNJQ1.png)](https://imgchr.com/i/DoNJQ1)
[![Doa0xA.png](https://s3.ax1x.com/2020/12/03/Doa0xA.png)](https://imgchr.com/i/Doa0xA)
 [![Doad8H.png](https://s3.ax1x.com/2020/12/03/Doad8H.png)](https://imgchr.com/i/Doad8H)
[![Doa7ZV.png](https://s3.ax1x.com/2020/12/03/Doa7ZV.png)](https://imgchr.com/i/Doa7ZV)
 [![Doa5Mn.png](https://s3.ax1x.com/2020/12/03/Doa5Mn.png)](https://imgchr.com/i/Doa5Mn)

>## 2.实现几何图形分类
#### 用卷积神经网络解决问题 vs 用卷积神经网络解决问题
* 卷积神经网络
  根据在测试集上得到的准确度是89.8%（测试过程在课件中有详细过程）
* 卷积神经网络 
  绝大部分样本预测是正确的，只有最后一个样本，看上去应该是一个很扁的三角形，被预测成了菱形，准确率非常高
> ## 3.MNIST分类
* 学习如何使用卷积网络来解决MNIST问题
#### 可视化
##### 第一组的卷积可视化

下图按行显示了以下内容：

1. 卷积核数值
2. 卷积核抽象
3. 卷积结果
4. 激活结果
5. 池化结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_layer_123_filter.png" ch="500" />
 卷积结果可视化

卷积核是5x5的，一共8个卷积核，所以第一行直接展示了卷积核的数值图形化以后的结果，但是由于色块太大，不容易看清楚其具体的模式，那么第二行的模式是如何抽象出来的呢？

因为特征是未知的，所以卷积神经网络不可能学习出类似下面的两个矩阵中左侧矩阵的整齐的数值，而很可能是如同右侧的矩阵一样具有很多噪音，但是大致轮廓还是个左上到右下的三角形，只是一些局部点上有一些值的波动。

```
2  2  1  1  0               2  0  1  1  0
2  1  1  0  0               2  1  1  2  0
1  1  0 -1 -2               0  1  0 -1 -2
1  0 -1 -2 -3               1 -1  1 -4 -3
0 -1 -2 -3 -4               0 -1 -2 -3 -2
```

 卷积核的抽象模式

|卷积核序号|1|2|3|4|5|6|7|8|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|抽象模式|右斜|下|中心|竖中|左下|上|右|左上|

这些模式实际上就是特征，是卷积网络自己学习出来的，每一个卷积核关注图像的一个特征，比如上部边缘、下部边缘、左下边缘、右下边缘等。这些特征的排列有什么顺序吗？没有。每一次重新训练后，特征可能会变成其它几种组合，顺序也会发生改变，这取决于初始化数值及样本顺序、批大小等等因素。

当然可以用更高级的图像处理算法，对5x5的图像进行模糊处理，再从中提取模式。

#### 第二组的卷积可视化

第二组的卷积、激活、池化层的输出结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_layer_456.png" ch="500" />

 第二组卷积核、激活、池化的可视化

- Conv2：由于是在第一层的特征图上卷积后叠加的结果，所以基本不能按照原图理解，但也能大致看出是是一些轮廓抽取的作用；
- Relu2：能看出的是如果黑色区域多的话，说明基本没有激活值，此卷积核效果就没用；
- Pool2：池化后分化明显的特征图是比较有用的特征，比如3、6、12、15、16；信息太多或者太少的特征图，都用途偏小，比如1、7、10、11。

> ## 5.Fashion-MNIST分类
#### 用前馈神经网络来解决问题 vs 用卷积神经网络来解决问题
* 与前馈神经网络方案相比，这32个样本里只有一个错误，第4行最后一列，把第9类“短靴”预测成了“凉鞋”，因为这个样本中间有一个三角形的黑色块，与凉鞋的镂空设计很像。
  ##### 使用卷积神经解决问题准确率更高。
>#### 代码及运行结果
* 前馈神经网络
[![DoDsN6.png](https://s3.ax1x.com/2020/12/03/DoDsN6.png)](https://imgchr.com/i/DoDsN6)
[![DoDinA.png](https://s3.ax1x.com/2020/12/03/DoDinA.png)](https://imgchr.com/i/DoDinA)
[![DoDwu9.png](https://s3.ax1x.com/2020/12/03/DoDwu9.png)](https://imgchr.com/i/DoDwu9)
* 卷积神经网络
[![DoDZh8.png](https://s3.ax1x.com/2020/12/03/DoDZh8.png)](https://imgchr.com/i/DoDZh8)
[![DoDN3F.png](https://s3.ax1x.com/2020/12/03/DoDN3F.png)](https://imgchr.com/i/DoDN3F)
[![DoD0BR.png](https://s3.ax1x.com/2020/12/03/DoD0BR.png)](https://imgchr.com/i/DoD0BR)
[![DoDBH1.png](https://s3.ax1x.com/2020/12/03/DoDBH1.png)](https://imgchr.com/i/DoDBH1)
> ## 6.Cifar-10分类
在CPU上训练

在CPU上训练，只设置了10个epoch，一共半个小时时间，在测试集上达到63.61%的准确率。观察val_loss和val_acc的趋势，随着训练次数的增加，还可以继续优化。

```
Epoch 1/10
1563/1563 [==============================] - 133s 85ms/step - loss: 1.8563 - acc: 0.3198 - val_loss: 1.5658 - val_acc: 0.4343
......

Epoch 10/10
1563/1563 [==============================] - 131s 84ms/step - loss: 1.0972 - acc: 0.6117 - val_loss: 1.0426 - val_acc: 0.6361

10000/10000 [==============================] - 7s 684us/step
Test loss: 1.042622245979309
Test accuracy: 0.6361
```
#### 代码及运行结果
[![DorVbR.png](https://s3.ax1x.com/2020/12/03/DorVbR.png)](https://imgchr.com/i/DorVbR)
[![DorEr9.png](https://s3.ax1x.com/2020/12/03/DorEr9.png)](https://imgchr.com/i/DorEr9)



>## 总结
卷积神经
通过再一次对这一章节内容的梳理，我对这一章节的内容有了新的理解：
1. 它的图层结构至少有以下四层卷积层、激活函数层、池化层、全连接分类层
2. 对于卷积我们需要注意到一维卷积指的是卷积核是1维的，而不是卷积的输入是1维的，1维指的是卷积方式。
3. 卷积的应用
   * 一维卷积常用于序列模型，自然语言处理领域。
   * 二维卷积常用于计算机视觉、图像处理领域。
4. 一般情况下，我们用正方形的卷积核，且为奇数,如果计算出的输出图片尺寸为小数，则取整，不做四舍五入
5. 卷积的前向运算在本章节他提到了三种方法：综合来说第三中最好，因为第一种运行慢，第二种不是所有python函数都适用，第三种综合一二种改良了。
6. 对于池化我们一般使用最大池化。 无论是max pooling还是mean pooling，都没有要学习的参数，所以，在卷积网络的训练中，池化层需要做的只是把误差项向后传递，不需要计算任何梯度。
7. 经典的卷积神经是先卷积然后再全连接
8. 色彩分类问题中使用卷积神经解决问题比前馈模型准确率大有提高,在运行python文件时，记住要先运行生成训练数据集的文件。


>## 心得体会
卷积神经网络的应用真的在我们生活中无处不在，人像物体等的识别都需要用到它，同时我意识到数学知识很重要，他的前向计算等等都需要数学公式的推导。卷积的运算也需要注意运行时间的把控，即使运算正确，但是速度太慢也是没有什么效率的。通过学习这一章节，我觉得好像前向卷积神经网络进行色彩，图像，物体的识别的准确率都比前馈神经网络的准确率要高。





