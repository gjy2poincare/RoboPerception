## 第17章 卷积神经网络原理
## 0 卷积神经网络概述

### 1 卷积神经网络的能力

卷积神经网络（CNN，Convolutional Neural Net)是神经网络的类型之一，在图像识别和分类领域中取得了非常好的效果，比如识别人脸、物体、交通标识等，这就为机器人、自动驾驶等应用提供了坚实的技术基础。
### 2 卷积神经网络的典型结构

一个典型的卷积神经网络的结构如图17-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_net.png" />

图17-5 卷积神经网络的典型结构图

在一个典型的卷积神经网络中，会至少包含以下几个层：

- 卷积层
- 激活函数层
- 池化层
- 全连接分类层
### 3 卷积核的作用
下面说明9个卷积核的作用：

表17-2 各个卷积核的作用

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

### 4 卷积后续的运算
前面我们认识到了卷积核的强大能力，卷积神经网络通过反向传播而令卷积核自我学习，找到分布在图片中的不同的feature，最后形成的卷积核中的数据。但是如果想达到这种效果，只有卷积层的话是不够的，还需要激活函数、池化等操作的配合。

图17-7中的四个子图，依次展示了：

1. 原图
2. 卷积结果
3. 激活结果
4. 池化结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/circle_conv_relu_pool.png" ch="500" />

图17-7 原图经过卷积-激活-池化操作后的效果

1. 注意图一是原始图片，用cv2读取出来的图片，其顺序是反向的，即：

- 第一维是高度
- 第二维是宽度
- 第三维是彩色通道数，但是其顺序为BGR，而不是常用的RGB

1. 我们对原始图片使用了一个3x1x3x3的卷积核，因为原始图片为彩色图片，所以第一个维度是3，对应RGB三个彩色通道；我们希望只输出一张feature map，以便于说明，所以第二维是1；我们使用了3x3的卷积核，用的是sobel x算子。所以图二是卷积后的结果。

2. 图三做了一层Relu激活计算，把小于0的值都去掉了，只留下了一些边的特征。

3. 图四是图三的四分之一大小，虽然图片缩小了，但是特征都没有丢失，反而因为图像尺寸变小而变得密集，亮点的密度要比图三大而粗。
### 5 卷积神经网络的学习
- 平移不变性
  
  对于原始图A，平移后得到图B，对于同一个卷积核来说，都会得到相同的特征，这就是卷积核的权值共享。但是特征处于不同的位置，由于距离差距较大，即使经过多层池化后，也不能处于近似的位置。此时，后续的全连接层会通过权重值的调整，把这两个相同的特征看作同一类的分类标准之一。如果是小距离的平移，通过池化层就可以处理了。

- 旋转不变性

  对于原始图A，有小角度的旋转得到C，卷积层在A图上得到特征a，在C图上得到特征c，可以想象a与c的位置间的距离不是很远，在经过两层池化以后，基本可以重合。所以卷积网络对于小角度旋转是可以容忍的，但是对于较大的旋转，需要使用数据增强来增加训练样本。一个极端的例子是当6旋转90度时，谁也不能确定它到底是6还是9。

- 尺度不变性

  对于原始图A和缩小的图D，人类可以毫不费力地辨别出它们是同一个东西。池化在这里是不是有帮助呢？没有！因为神经网络对A做池化的同时，也会用相同的方法对D做池化，这样池化的次数一致，最终D还是比A小。如果我们有多个卷积视野，相当于从两米远的地方看图A，从一米远的地方看图D，那么A和D就可以很相近似了。这就是Inception的想法，用不同尺寸的卷积核去同时寻找同一张图片上的特征。
  ## 1 卷积的前向计算

### 1 卷积的数学定义

#### 连续定义

$$h(x)=(f*g)(x) = \int_{-\infty}^{\infty} f(t)g(x-t)dt \tag{1}$$

卷积与傅里叶变换有着密切的关系。利用这点性质，即两函数的傅里叶变换的乘积等于它们卷积后的傅里叶变换，能使傅里叶分析中许多问题的处理得到简化。

#### 离散定义

$$h(x) = (f*g)(x) = \sum^{\infty}_{t=-\infty} f(t)g(x-t) \tag{2}$$
## 2 一维卷积实例

有两枚骰子$f,g$，掷出后二者相加为4的概率如何计算？

第一种情况：$f(1)g(3), 3+1=4$，如图17-9所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/touzi1.png" />

图17-9 第一种情况

第二种情况：$f(2)g(2), 2+2=4$，如图17-10所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/touzi2.png" />

图17-10 第二种情况

第三种情况：$f(3)g(1), 1+3=4$，如图17-11所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/touzi3.png" />

图17-11 第三种情况

因此，两枚骰子点数加起来为4的概率为：

$$
\begin{aligned}
h(4) &= f(1)g(3)+f(2)g(2)+f(3)g(1) \\\\
&=f(1)g(4-1) + f(2)g(4-2) + f(3)g(4-3)
\end{aligned}
$$

符合卷积的定义，把它写成标准的形式就是公式2：

$$h(4)=(f*g)(4)=\sum _{t=1}^{3}f(t)g(4-t)$$

### 3 单入单出的二维卷积
1. 我们实现的卷积操作不是原始数学含义的卷积，而是工程上的卷积，可以简称为卷积
2. 在实现卷积操作时，并不会反转卷积核
在传统的图像处理中，卷积操作多用来进行滤波，锐化或者边缘检测啥的。我们可以认为卷积是利用某些设计好的参数组合（卷积核）去提取图像空域上相邻的信息。

按照公式5，我们可以在4x4的图片上，用一个3x3的卷积核，通过卷积运算得到一个2x2的图片，运算的过程如图17-12所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_w3_s1.png" ch="526" />

图17-12 卷积运算的过程

### 4 单入多出的升维卷积

原始输入是一维的图片，但是我们可以用多个卷积核分别对其计算，从而得到多个特征输出。如图17-13所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_2w3.png" ch="500" />

图17-13 单入多出的升维卷积

一张4x4的图片，用两个卷积核并行地处理，输出为2个2x2的图片。在训练过程中，这两个卷积核会完成不同的特征学习。

### 5 多入单出的降维卷积
一张图片，通常是彩色的，具有红绿蓝三个通道。我们可以有两个选择来处理：

1. 变成灰度的，每个像素只剩下一个值，就可以用二维卷积
2. 对于三个通道，每个通道都使用一个卷积核，分别处理红绿蓝三种颜色的信息

显然第2种方法可以从图中学习到更多的特征，于是出现了三维卷积，即有三个卷积核分别对应书的三个通道，三个子核的尺寸是一样的，比如都是2x2，这样的话，这三个卷积核就是一个3x2x2的立体核，称为过滤器Filter，所以称为三维卷积。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/multiple_filter.png" />
### 6 多入多出的同维卷积

在上面的例子中，是一个过滤器Filter内含三个卷积核Kernal。我们假设有一个彩色图片为3x3的，如果有两组3x2x2的卷积核的话，会做什么样的卷积计算？看图17-15。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv3dp.png" ch="500" />

图17-15 多入多出的卷积运算

第一个过滤器Filter-1为棕色所示，它有三卷积核(Kernal)，命名为Kernal-1，Keanrl-2，Kernal-3，分别在红绿蓝三个输入通道上进行卷积操作，生成三个2x2的输出Feature-1,n。然后三个Feature-1,n相加，并再加上b1偏移值，形成最后的棕色输出Result-1。

对于灰色的过滤器Filter-2也是一样，先生成三个Feature-2,n，然后相加再加b2，最后得到Result-2。

之所以Feature-m,n还用红绿蓝三色表示，是因为在此时，它们还保留着红绿蓝三种色彩的各自的信息，一旦相加后得到Result，这种信息就丢失了。
### 7 卷积编程模型

上图侧重于解释数值计算过程，而图17-16侧重于解释五个概念的关系：

- 输入 Input Channel
- 卷积核组 WeightsBias
- 过滤器 Filter
- 卷积核 kernal
- 输出 Feature Map

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv3d.png" ch="500" />

图17-16 三通道经过两组过滤器的卷积过程

在此例中，输入是三维数据（3x32x32），经过2x3x5x5的卷积后，输出为三维（2x28x28），维数并没有变化，只是每一维内部的尺寸有了变化，一般都是要向更小的尺寸变化，以便于简化计算。

对于三维卷积，有以下特点：

1. 预先定义输出的feature map的数量，而不是根据前向计算自动计算出来，此例中为2，这样就会有两组WeightsBias
2. 对于每个输出，都有一个对应的过滤器Filter，此例中Feature Map-1对应Filter-1
3. 每个Filter内都有一个或多个卷积核Kernal，对应每个输入通道(Input Channel)，此例为3，对应输入的红绿蓝三个通道
4. 每个Filter只有一个Bias值，Filter-1对应b1，Filter-2对应b2
5. 卷积核Kernal的大小一般是奇数如：1x1, 3x3, 5x5, 7x7等，此例为5x5

对于上图，我们可以用在全连接神经网络中的学到的知识来理解：

1. 每个Input Channel就是特征输入，在上图中是3个
2. 卷积层的卷积核相当于隐层的神经元，上图中隐层有2个神经元
3. $W(m,n), m=[1,2], n=[1,3]$相当于隐层的权重矩阵$w_{11},w_{12},......$
4. 每个卷积核（神经元）有1个偏移值

### 8 步长 stride

前面的例子中，每次计算后，卷积核会向右或者向下移动一个单元，即步长stride = 1。而在图17-17这个卷积操作中，卷积核每次向右或向下移动两个单元，即stride = 2。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/Stride2.png" />

图17-17 步长为2的卷积

在后续的步骤中，由于每次移动两格，所以最终得到一个2x2的图片。

### 9 填充 padding

如果原始图为4x4，用3x3的卷积核进行卷积后，目标图片变成了2x2。如果我们想保持目标图片和原始图片为同样大小，该怎么办呢？一般我们会向原始图片周围填充一圈0，然后再做卷积。如图17-18。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/padding.png" ch="500" />

图17-18 带填充的卷积
## 2 卷积前向计算代码实现

###  1 卷积核的实现
卷积核，实际上和全连接层一样，是权重矩阵加偏移向量的组合，区别在于全连接层中的权重矩阵是二维的，偏移矩阵是列向量，而卷积核的权重矩阵是四维的，偏移矩阵是也是列向量。

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

图17-19 卷积核的组成

以图17-19为例，各个维度的数值如下：

- FilterCount=2，第一维，过滤器数量，对应输出通道数。
- KernalCount=3，第二维，卷积核数量，对应输入通道数。两个Filter里面的Kernal数必须相同。
- KernalHeight=5，KernalWidth=5，卷积核的尺寸，第三维和第四维。同一组WeightsBias里的卷积核尺寸必须相同。

在初始化函数中，会根据四个参数定义`WBShape`，然后在`CreateNew`函数中，创建相应形状的`Weights`和`Bias`。
### 2 卷积前向运算的实现 
```Python
class ConvLayer(CLayer):
    def forward(self, x, train=True):
        self.x = x
        self.batch_size = self.x.shape[0]
        # 如果有必要的话，先对输入矩阵做padding
        if self.padding > 0:
            self.padded = np.pad(...)
        else:
            self.padded = self.x
        #end if
        self.z = conv_4d(...)
        return self.z
```

上述代码中的`conv_4d()`函数实现了17.1中定义的四维卷积运算：

```Python
def conv_4d(x, weights, bias, out_h, out_w, stride=1):
    batch_size = x.shape[0]
    input_channel = x.shape[1]
    output_channel = weights.shape[0]
    filter_height = weights.shape[2]
    filter_width = weights.shape[3]
    rs = np.zeros((batch_size, num_output_channel, out_h, out_w))

    for bs in range(batch_size):
        for oc in range(output_channel):
            rs[bs,oc] += bias[oc]
            for ic in range(input_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        ii = i * stride
                        jj = j * stride
                        for fh in range(filter_height):
                            for fw in range(filter_width):
                                rs[bs,oc,i,j] += x[bs,ic,fh+ii,fw+jj] * weights[oc,ic,fh,fw]
```

上面的函数包含以下几重循环：

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

出乎我们的预料，在足足等了30多秒后，才返回结果。

通过试验发现，其运行速度非常慢，如果这样的函数在神经网络训练中被调用几十万次，其性能是非常糟糕的，这也是`Python`做为动态语言的一个缺点。
## 3 卷积层的训练
### 1 计算反向传播的梯度矩阵

正向公式：

$$Z = W*A+b \tag{0}$$

其中，W是卷积核，*表示卷积（互相关）计算，A为当前层的输入项，b是偏移（未在图中画出），Z为当前层的输出项，但尚未经过激活函数处理。

我们举一个具体的例子便于分析。图17-21是正向计算过程。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_forward.png" />

图17-21 卷积正向运算
我们把传入的误差矩阵Delta-In做一个zero padding，再乘以旋转180度的卷积核，就是要传出的误差矩阵Delta-Out，如图17-22所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_backward.png" />

图17-22 卷积运算中的误差反向传播

最后可以统一成为一个简洁的公式：

$$\delta_{out} = \delta_{in} * W^{rot180} \tag{8}$$

这个误差矩阵可以继续回传到下一层。

- 当Weights是$3\times 3$时，$\delta_{in}$需要padding=2，即加2圈0，才能和Weights卷积后，得到正确尺寸的$\delta_{out}$
- 当Weights是$5\times 5$时，$\delta_{in}$需要padding=4，即加4圈0，才能和Weights卷积后，得到正确尺寸的$\delta_{out}$
- 以此类推：当Weights是$N\times N$时，$\delta_{in}$需要padding=N-1，即加N-1圈0。
### 2 步长不为1时的梯度矩阵还原

我们先观察一下stride=1和2时，卷积结果的差异如图17-23。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/stride_1_2.png"/>

图17-23 步长为1和步长为2的卷积结果的比较

二者的差别就是中间那个结果图的灰色部分。如果反向传播时，传入的误差矩阵是stride=2时的2x2的形状，那么我们只需要把它补上一个十字，变成3x3的误差矩阵，就可以用步长为1的算法了。

以此类推，如果步长为3时，需要补一个双线的十字。所以，当知道当前的卷积层步长为S（S>1）时：

1. 得到从上层回传的误差矩阵形状，假设为$M \times N$
2. 初始化一个$(M \cdot S) \times (N \cdot S)$的零矩阵
3. 把传入的误差矩阵的第一行值放到零矩阵第0行的0,S,2S,3S...位置
4. 然后把误差矩阵的第二行的值放到零矩阵第S行的0,S,2S,3S...位置
5. ......
### 3 有多个卷积核时的梯度计算

有多个卷积核也就意味着有多个输出通道。

也就是14.1中的升维卷积，如图17-24。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_2w2.png" ch="500" />

图17-24 升维卷积
$$\delta_{out} = \sum_m \delta_{in\_m} * W^{rot180}_ m \tag{9}$$

### 4 有多个输入时的梯度计算

当输入层是多个图层时，每个图层必须对应一个卷积核，如图17-25。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_1W222.png" ch="500" />

图17-25 多个图层的卷积必须有一一对应的卷积核
$$\delta_{out2} = \delta_{in} * W_2^{rot180} \tag{15}$$

### 5 权重（卷积核）梯度计算

图17-26展示了我们已经熟悉的卷积正向运算。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_forward.png" />

图17-26 卷积正向计算
$$\delta_w = A * \delta_{in} \tag{11}$$

### 6 偏移的梯度计算

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
## 4 卷积反向传播代码实现

### 1. 方法1

完全按照17.3中的讲解来实现反向传播，但是由于有17.2中关于numba帮助，我们在实现代码时，可以考虑把一些模块化的计算放到独立的函数中，用numba在运行时编译加速。

```Python
    def backward_numba(self, delta_in, flag):
        # 如果正向计算中的stride不是1，转换成是1的等价误差数组
        dz_stride_1 = expand_delta_map(delta_in, ...)
        # 计算本层的权重矩阵的梯度
        self._calculate_weightsbias_grad(dz_stride_1)
        # 由于输出误差矩阵的尺寸必须与本层的输入数据的尺寸一致，所以必须根据卷积核的尺寸，调整本层的输入误差矩阵的尺寸
        (pad_h, pad_w) = calculate_padding_size(...)
        dz_padded = np.pad(dz_stride_1, ...)
        # 计算本层输出到下一层的误差矩阵
        delta_out = self._calculate_delta_out(dz_padded, flag)
        #return delta_out
        return delta_out, self.WB.dW, self.WB.dB

    # 用输入数据乘以回传入的误差矩阵,得到卷积核的梯度矩阵
    def _calculate_weightsbias_grad(self, dz):
        self.WB.ClearGrads()
        # 先把输入矩阵扩大，周边加0
        (pad_h, pad_w) = calculate_padding_size(...)
        input_padded = np.pad(self.x, ...)
        # 输入矩阵与误差矩阵卷积得到权重梯度矩阵
        (self.WB.dW, self.WB.dB) = calcalate_weights_grad(...)
        self.WB.MeanGrads(self.batch_size)

    # 用输入误差矩阵乘以（旋转180度后的）卷积核
    def _calculate_delta_out(self, dz, layer_idx):
        if layer_idx == 0:
            return None
        # 旋转卷积核180度
        rot_weights = self.WB.Rotate180()
        # 定义输出矩阵形状
        delta_out = np.zeros(self.x.shape)
        # 输入梯度矩阵卷积旋转后的卷积核，得到输出梯度矩阵
        delta_out = calculate_delta_out(dz, ..., delta_out)

        return delta_out
```
两个计算量大的函数，一个是计算权重矩阵的基础函数calcalate_weights_grad，另一个是计算输出误差矩阵的基础函数calculate_delta_out，都使用了numba的方式实现，以加快反向传播代码的运行速度。

### 2 方法2
#### 代码实现

```Python
    def backward_col2img(self, delta_in, layer_idx):
        OutC, InC, FH, FW = self.WB.W.shape
        # 误差矩阵变换
        delta_in_2d = np.transpose(delta_in, axes=(0,2,3,1)).reshape(-1, OutC)
        # 计算Bias的梯度
        self.WB.dB = np.sum(delta_in_2d, axis=0, keepdims=True).T / self.batch_size
        # 计算Weights的梯度
        dW = np.dot(self.col_x.T, delta_in_2d) / self.batch_size
        # 转换成卷积核的原始形状
        self.WB.dW = np.transpose(dW, axes=(1, 0)).reshape(OutC, InC, FH, FW)# 计算反向传播误差矩阵
        dcol = np.dot(delta_in_2d, self.col_w.T)
        # 转换成与输入数据x相同的形状
        delta_out = col2img(dcol, self.x.shape, FH, FW, self.stride, self.padding)
        return delta_out, self.WB.dW, self.WB.dB
```

#### 单样本单通道的实例讲解

假设有1个样本1个通道且图片为3x3的矩阵：
```
x=
 [[[[0 1 2]
    [3 4 5]
    [6 7 8]]]]

col_x=
 [[0. 1. 3. 4.]
  [1. 2. 4. 5.]
  [3. 4. 6. 7.]
  [4. 5. 7. 8.]]
```
卷积核也只有1个形状为1x1x2x2的矩阵：
```
w=
 [[[[0 1]
    [2 3]]]]
```
卷积核展开后：
```
col_w=
 [[0]
  [1]
  [2]
  [3]]
```
卷积的结果会是一个样本在一个通道上的2x2的输出。

再假设从后端反向传播回来的输入误差矩阵：
```
delta_in=
 [[[[0 1]
    [2 3]]]]
```
误差矩阵经过下式变换：
```Python
delta_in_2d = np.transpose(delta_in, axes=(0,2,3,1)).reshape(-1, OutC)
```
得到：
```
delta_in_2d=
 [[0]
 [1]
 [2]
 [3]]
```
计算dB（这一步和全连接层完全相同）：
```Python
self.WB.dB = np.sum(delta_in_2d, axis=0, keepdims=True).T / self.batch_size
```
得到：
```
dB=
 [[6.]]
```
计算dW（这一步和全连接层完全相同）：
```Python
dW = np.dot(self.col_x.T, delta_in_2d) / self.batch_size
```
得到：
```
dW=
 [[19.]
  [25.]
  [37.]
  [43.]]
```
还原dW到1x1x2x2的卷积核形状：
```Python
self.WB.dW = np.transpose(dW, axes=(1, 0)).reshape(OutC, InC, FH, FW) 
```
得到：
```
dW=
 [[[[19. 25.]
    [37. 43.]]]]
```
至此，dB和dW都已经得到，本层的梯度计算完毕，需要把梯度回传给前一层，所以要计算输出误差矩阵（这一步和全连接层完全相同）：
```Python
dcol = np.dot(delta_in_2d, self.col_w.T)
```
得到：
```
dcol=
 [[0 0 0 0]
 [0 1 2 3]
 [0 2 4 6]
 [0 3 6 9]]
```
转换成正确的矩阵形状：
```Python
delta_out = col2img(dcol, self.x.shape, FH, FW, self.stride, self.padding)
```
得到：
```
delta_out=
 [[[[ 0.  0.  1.]
    [ 0.  4.  6.]
    [ 4. 12.  9.]]]]
```

下面我们解释一下最后一步的运算过程，如图17-30所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/col2img.png" ch="500" />

图17-30 col2img图解

得到上述dcol的4x4矩阵后，我们要把它逆变换到一个3x3的矩阵中，步骤如下：

1. 左侧第一行红色椭圆内的四个元素移到右侧红色圆形内；
2. 在1的基础上，左侧第二行黄色椭圆内的四个元素移到右侧黄色圆形内，其中与原有元素重叠的地方则两个值相加。比如中间那个元素就是0+2=2；
3. 在2的基础上，左侧第三行蓝色椭圆内的四个元素移到右侧蓝色圆形内，其中与原有元素重叠的地方则两个值相加。比如中间那个元素再次加2；
4. 在3的基础上，左侧第四行绿色椭圆内的四个元素移到右侧绿色圆形内，其中与原有元素重叠的地方则两个值相加，中间的元素再次加0，还是4；中间靠下的元素原值是6，加6后为12。

这个结果和最后一步delta_out的结果完全一致。

#### 多样本多通道的实例讲解

图17-31是两个样本的例子，输入通道为3，输出通道为2。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv4d.png" ch="500" />

图17-31 两个样本三通道两个卷积核的例子

图17-31中的各组件参数如下：

- batch size = 2
- input channel = 3
- input height = 3
- input width = 3
- filter height = 2
- filter width = 2
- stride = 1
- padding = 0
- output channel = 2
- output height = 2
- output width = 2

#### 误差输入矩阵

delta_in是本层的误差输入矩阵，它的形状应该和本层的前向计算结果一样。在本例中，误差输入矩阵的形状应该是：(batch_size * output_channel * output_height * output_width) = (2 x 2 x 2 x 2)：
```
delta_in=
(样本1)
    (通道1)
 [[[[ 0  1]
   [ 2  3]]
    (通道2)
  [[ 4  5]
   [ 6  7]]]
(样本2)
    (通道1)
 [[[ 8  9]
   [10 11]]
    (通道2)
  [[12 13]
   [14 15]]]]
```

为了做img2col的逆运算col2img，我们把它转换成17.2中的结果数据的形状8x2：

```Python
delta_in_2d = np.transpose(delta_in, axes=(0,2,3,1)).reshape(-1, OutC)
```

```
delta_in_2d=
 [[ 0  4]
  [ 1  5]
  [ 2  6]
  [ 3  7]
  [ 8 12]
  [ 9 13]
  [10 14]
  [11 15]]
```

计算权重矩阵的梯度：

```Python
dW = np.dot(self.col_x.T, delta_in_2d) / self.batch_size
```

结果：

```
dW=
[[ 564.|  812.]
 [ 586.|  850.]
 [ 630.|  926.]
 [ 652.|  964.]
 ------+-------
 [ 762.| 1154.]
 [ 784.| 1192.]
 [ 828.| 1268.]
 [ 850.| 1306.]
 ------+-------
 [ 960.| 1496.]
 [ 982.| 1534.]
 [1026.| 1610.]
 [1048.| 1648.]]
```
但是这个12x2的结果是对应的权重矩阵的二维数组展开形式的，所以要还原成原始的卷积核形式2x3x2x2：

```Python
self.WB.dW = np.transpose(dW, axes=(1, 0)).reshape(OutC, InC, FH, FW)
```

结果：

```
dW=
(过滤器1)                 (过滤器2)
    (卷积核1)                 (卷积核1)
 [[[[ 564.  586.]        [[[ 812.  850.]
   [ 630.  652.]]          [ 926.  964.]]
    (卷积核2)                 (卷积核2)
  [[ 762.  784.]          [[1154. 1192.]
   [ 828.  850.]]          [1268. 1306.]]
    (卷积核3)                 (卷积核3)
  [[ 960.  982.]          [[1496. 1534.]
   [1026. 1048.]]]         [1610. 1648.]]]]
```

计算误差输出矩阵：

```Python
dcol = np.dot(delta_in_2d, self.col_w.T)
```

得到：

```
dcol=
 [[ 48  52  56  60  64  68  72  76  80  84  88  92]
 [ 60  66  72  78  84  90  96 102 108 114 120 126]
 [ 72  80  88  96 104 112 120 128 136 144 152 160]
 [ 84  94 104 114 124 134 144 154 164 174 184 194]
 [144 164 184 204 224 244 264 284 304 324 344 364]
 [156 178 200 222 244 266 288 310 332 354 376 398]
 [168 192 216 240 264 288 312 336 360 384 408 432]
 [180 206 232 258 284 310 336 362 388 414 440 466]]
```

但是dcol对应的是输入数据的二维展开形式4x12，应该把它还原成2x3x3x3的形式：

```Python
delta_out = col2img(dcol, self.x.shape, FH, FW, self.stride, self.padding)
```

得到：

```
delta_out=
(样本1)                     (样本2)
    (通道1)                     (通道1)
[[[[  48.  112.   66.]    [[[ 144.  320.  178.]
   [ 128.  296.  172.]      [ 352.  776.  428.]
   [  88.  200.  114.]]     [ 216.  472.  258.]]
    (通道2)                     (通道2)
  [[  64.  152.   90.]     [[ 224.  488.  266.]
   [ 176.  408.  236.]      [ 528. 1144.  620.]
   [ 120.  272.  154.]]     [ 312.  672.  362.]]
    (通道3)                     (通道3)
  [[  80.  192.  114.]     [[ 304.  656.  354.]
   [ 224.  520.  300.]      [ 704. 1512.  812.]
   [ 152.  344.  194.]]]    [ 408.  872.  466.]]]]
```

## 5 池化层

### 1 常用池化方法

池化 pooling，又称为下采样，downstream sampling or sub-sampling。

池化方法分为两种，一种是最大值池化 Max Pooling，一种是平均值池化 Mean/Average Pooling。如图17-32所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/pooling.png" />

图17-32 池化

- 最大值池化，是取当前池化视野中所有元素的最大值，输出到下一层特征图中。
- 平均值池化，是取当前池化视野中所有元素的平均值，输出到下一层特征图中。

其目的是：

- 扩大视野：就如同先从近处看一张图片，然后离远一些再看同一张图片，有些细节就会被忽略
- 降维：在保留图片局部特征的前提下，使得图片更小，更易于计算
- 平移不变性，轻微扰动不会影响输出：比如上图中最大值池化的4，即使向右偏一个像素，其输出值仍为4
- 维持同尺寸图片，便于后端处理：假设输入的图片不是一样大小的，就需要用池化来转换成同尺寸图片

一般我们都使用最大值池化。

### 2 池化的其它方式

在上面的例子中，我们使用了size=2x2，stride=2的模式，这是常用的模式，即步长与池化尺寸相同。

我们很少使用步长值与池化尺寸不同的配置，所以只是提一下，如图17-33。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/pooling2.png" />

图17-33 步长为1的池化

上图是stride=1, size=2x2的情况，可以看到，右侧的结果中，有一大堆的3和4，基本分不开了，所以其池化效果并不好。

假设输入图片的形状是 $W_1 \times H_1 \times D_1$，其中W是图片宽度，H是图片高度，D是图片深度（多个图层），F是池化的视野（正方形），S是池化的步长，则输出图片的形状是：

$$
\begin{cases}
W_2 = (W_1 - F)/S + 1 \\\\
H_2 = (H_1 - F)/S + 1 \\\\
D_2 = D_1
\end{cases}
$$

池化层不会改变图片的深度，即D值前后相同。

### 3 池化层的训练

我们假设图17-34中，$[[1,2],[3,4]]$是上一层网络回传的残差，那么：

- 对于最大值池化，残差值会回传到当初最大值的位置上，而其它三个位置的残差都是0。
- 对于平均值池化，残差值会平均到原始的4个位置上。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/pooling_backward.png" />

图17-34 平均池化与最大池化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/pooling_backward_max.png" />

图17-35 池化层反向传播的示例

#### Max Pooling

严格的数学推导过程以图17-35为例进行。

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

#### Mean Pooling

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
# 第18章 卷积神经网络应用

## 0 经典的卷积神经网络模型
### 1 LeNet (1998)

图18-1是卷积神经网络的鼻祖LeNet$^{[3]}$的模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_LeNet.png" />

图18-1 LeNet模型结构图

LeNet是卷积神经网络的开创者LeCun在1998年提出，用于解决手写数字识别的视觉任务。自那时起，卷积神经网络的最基本的架构就定下来了：卷积层、池化层、全连接层。

1. 输入为单通道32x32灰度图
2. 使用6组5x5的过滤器，每个过滤器里有一个卷积核，stride=1，得到6张28x28的特征图
3. 使用2x2的池化，stride=2，得到6张14x14的特征图
4. 使用16组5x5的过滤器，每个过滤器里有6个卷积核，对应上一层的6个特征图，得到16张10x10的特征图
5. 池化，得到16张5x5的特征图
6. 接全连接层，120个神经元
7. 接全连接层，84个神经元
8. 接全连接层，10个神经元，softmax输出
### 2 AlexNet (2012)

AlexNet$^{[4]}$网络结构在整体上类似于LeNet，都是先卷积然后在全连接。但在细节上有很大不同。AlexNet更为复杂。AlexNet有60 million个参数和65000个神经元，五层卷积，三层全连接网络，最终的输出层是1000通道的Softmax。

AlexNet用两块GPU并行计算，大大提高了训练效率，并且在ILSVRC-2012竞赛中获得了top-5测试的15.3%的error rate，获得第二名的方法error rate是26.2%，差距非常大，足以说明这个网络在当时的影响力。

图18-2是AlexNet的模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_AlexNet.png" />

图18-2 AlexNet模型结构图

1. 原始图片是3x224x224三通道彩色图片，分开到上下两块GPU上进行训练
2. 卷积，96核，11x11，stride=4，pad=0，输出96x55x55
3. LRN+池化，3x3，stride=2，pad=0，输出96x27x27
4. 卷积，256核，5x5，stride=1，pad=2，输出256x27x27
5. LRN+池化，3x3，stride=2，输出256x13x13
6. 卷积，384核，3x3，stride=1，pad=1，输出384x13x13
7. 卷积，384核，3x3，stride=1，pad=1，输出384x13x13
8. 卷积，256核，3x3，stride=1，pad=1，输出256x13x13
9. 池化，3x3，stride=2，输出256x6x6
10. 全连接层，4096个神经元，接Dropout和Relu
11. 全连接层，4096个神经元，接Dropout和Relu
12. 全连接层，1000个神经元做分类

AlexNet的特点：

- 比LeNet深和宽的网络
  
  使用了5层卷积和3层全连接，一共8层。特征数在最宽处达到384。

- 数据增强
  
  针对原始图片256x256的数据，做了随机剪裁，得到224x224的图片若干张。

- 使用ReLU做激活函数
- 在全连接层使用DropOut
- 使用LRN
  
  LRN的全称为Local Response Normalizatio，局部响应归一化，是想对线性输出做一个归一化，避免上下越界。发展至今，这个技术已经很少使用了。

### 3 ZFNet (2013)
图18-3是ZFNet的结构示意图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ZFNet.png" />

图18-3 ZFNet模型结构图

但是ZFNet首次系统化地对卷积神经网络做了可视化的研究，从而找到了AlexNet的缺点并加以改正，提高了网络的能力。总的来说，通过卷积神经网络学习后，我们学习到的特征，是具有辨别性的特征，比如要我们区分人脸和狗头，那么通过卷积神经网络学习后，背景部位的激活度基本很少，我们通过可视化就可以看到我们提取到的特征忽视了背景，而是把关键的信息给提取出来了。
### 4 VGGNet (2015)
图18-7为VGG16（16层的VGG）模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_Vgg16.png" ch="500" />

图18-7 VGG16模型结构图

VGG比较出名的是VGG-16和VGG-19，最常用的是VGG-16。

VGGNet的卷积层有一个特点：特征图的空间分辨率单调递减，特征图的通道数单调递增，使得输入图像在维度上流畅地转换到分类向量。用通俗的语言讲，就是特征图尺寸单调递减，特征图数量单调递增。从上面的模型图上来看，立体方块的宽和高逐渐减小，但是厚度逐渐增加。

AlexNet的通道数无此规律，VGGNet后续的GoogLeNet和Resnet均遵循此维度变化的规律。

一些其它的特点如下：

1. 选择采用3x3的卷积核是因为3x3是最小的能够捕捉像素8邻域信息的尺寸。
2. 使用1x1的卷积核目的是在不影响输入输出的维度情况下，对输入进行形变，再通过ReLU进行非线性处理，提高决策函数的非线性。
3. 2个3x3卷积堆叠等于1个5x5卷积，3个3x3堆叠等于1个7x7卷积，感受野大小不变，而采用更多层、更小的卷积核可以引入更多非线性（更多的隐藏层，从而带来更多非线性函数），提高决策函数判决力，并且带来更少参数。
4. 每个VGG网络都有3个FC层，5个池化层，1个softmax层。
5. 在FC层中间采用dropout层，防止过拟合。

虽然 VGGNet 减少了卷积层参数，但实际上其参数空间比 AlexNet 大，其中绝大多数的参数都是来自于第一个全连接层，耗费更多计算资源。在随后的 NIN 中发现将这些全连接层替换为全局平均池化，对于性能影响不大，同时显著降低了参数数量。

采用 Pre-trained 方法训练的 VGG model（主要是 D 和 E），相对其他的方法参数空间很大，所以训练一个 VGG 模型通常要花费更长的时间，所幸有公开的 Pre-trained model 让我们很方便的使用。

### 5 GoogLeNet (2014)
图18-8是GoogLeNet的模型结构图。
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_GoogleNet.png" />

图18-8 GoogLeNet模型结构图

蓝色为卷积运算，红色为池化运算，黄色为softmax分类。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_GoogleNet_Inception.png" />

图18-9 Inception结构图

图18-9的结构就是Inception，结构里的卷积stride都是1，另外为了保持特征响应图大小一致，都用了零填充。最后每个卷积层后面都立刻接了个ReLU层。在输出前有个叫concatenate的层，直译的意思是“并置”，即把4组不同类型但大小相同的特征响应图一张张并排叠起来，形成新的特征响应图。Inception结构里主要做了两件事：1. 通过3×3的池化、以及1×1、3×3和5×5这三种不同尺度的卷积核，一共4种方式对输入的特征响应图做了特征提取。2. 为了降低计算量。同时让信息通过更少的连接传递以达到更加稀疏的特性，采用1×1卷积核来实现降维。

这里想再详细谈谈1×1卷积核的作用，它究竟是怎么实现降维的。现在运算如下：下面图1是3×3卷积核的卷积，图2是1×1卷积核的卷积过程。对于单通道输入，1×1的卷积确实不能起到降维作用，但对于多通道输入，就不同了。假设你有256个特征输入，256个特征输出，同时假设Inception层只执行3×3的卷积。这意味着总共要进行 256×256×3×3的卷积（589000次乘积累加（MAC）运算）。这可能超出了我们的计算预算，比方说，在Google服务器上花0.5毫秒运行该层。作为替代，我们决定减少需要卷积的特征的数量，比如减少到64（256/4）个。在这种情况下，我们首先进行256到64的1×1卷积，然后在所有Inception的分支上进行64次卷积，接着再使用一个64到256的1×1卷积。

- 256×64×1×1 = 16000
- 64×64×3×3 = 36000
- 64×256×1×1 = 16000

现在的计算量大约是70000(即16000+36000+16000)，相比之前的约600000，几乎减少了10倍。这就通过小卷积核实现了降维。
现在再考虑一个问题：为什么一定要用1×1卷积核，3×3不也可以吗？考虑[50,200,200]的矩阵输入，我们可以使用20个1×1的卷积核进行卷积，得到输出[20,200,200]。有人问，我用20个3×3的卷积核不是也能得到[20,200,200]的矩阵输出吗，为什么就使用1×1的卷积核？我们计算一下卷积参数就知道了，对于1×1的参数总数：20×200×200×（1×1），对于3×3的参数总数：20×200×200×（3×3），可以看出，使用1×1的参数总数仅为3×3的总数的九分之一！所以我们使用的是1×1卷积核。

GoogLeNet网络结构中有3个LOSS单元，这样的网络设计是为了帮助网络的收敛。在中间层加入辅助计算的LOSS单元，目的是计算损失时让低层的特征也有很好的区分能力，从而让网络更好地被训练。在论文中，这两个辅助LOSS单元的计算被乘以0.3，然后和最后的LOSS相加作为最终的损失函数来训练网络。

GoogLeNet还有一个闪光点值得一提，那就是将后面的全连接层全部替换为简单的全局平均pooling，在最后参数会变的更少。而在AlexNet中最后3层的全连接层参数差不多占总参数的90%，使用大网络在宽度和深度允许GoogleNet移除全连接层，但并不会影响到结果的精度，在ImageNet中实现93.3%的精度，而且要比VGG还要快。

### 6 ResNets (2015)
图18-10是ResNets的模型结构。
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ResNet.png" />

图18-10 ResNets模型结构图

ResNet称为残差网络。什么是残差呢？

若将输入设为X，将某一有参网络层设为H，那么以X为输入的此层的输出将为H(X)。一般的卷积神经网络网络如Alexnet/VGG等会直接通过训练学习出参数函数H的表达，从而直接学习X -> H(X)。
而残差学习则是致力于使用多个有参网络层来学习输入、输出之间的参差即H(X) - X即学习X -> (H(X) - X) + X。其中X这一部分为直接的identity mapping，而H(X) - X则为有参网络层要学习的输入输出间残差。

图18-11为残差学习这一思想的基本表示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ResNet_flow.png" />

图18-11 残差结构示意图

图18-12展示了两种形态的残差模块，左图是常规残差模块，有两个3×3卷积核组成，但是随着网络进一步加深，这种残差结构在实践中并不是十分有效。针对这问题，右图的“瓶颈残差模块”（bottleneck residual block）可以有更好的效果，它依次由1×1、3×3、1×1这三个卷积层堆积而成，这里的1×1的卷积能够起降维或升维的作用，从而令3×3的卷积可以在相对较低维度的输入上进行，以达到提高计算效率的目的。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ResNet_block.png" />

图18-12 两种形态的残差模块

### 7 DenseNet (2017)

DenseNet$^{[9]}$ 是一种具有密集连接的卷积神经网络。在该网络中，任何两层之间都有直接的连接，也就是说，网络每一层的输入都是前面所有层输出的并集，而该层所学习的特征图也会被直接传给其后面所有层作为输入。下图是 DenseNet 的一个dense block示意图，一个block里面的结构如下，与ResNet中的BottleNeck基本一致：BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) ，而一个DenseNet则由多个这种block组成。每个DenseBlock的之间层称为transition layers，由BN−>Conv(1×1)−>averagePooling(2×2)组成。

图18-13是ResNets的模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_DenseNet_1.png" ch="500" />

图18-13 DenseNet模型结构图

DenseNet作为另一种拥有较深层数的卷积神经网络，具有如下优点：

1. 相比ResNet拥有更少的参数数量
2. 旁路加强了特征的重用
3. 网络更易于训练,并具有一定的正则效果
4. 缓解了gradient vanishing和model degradation的问题
## 1 实现颜色分类
### 搭建模型

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

#### 运行结果

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

在图18-16的可视化结果，一共64张图，是测试集中1000个样本的前64个样本，每张图上方的标签是预测的结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/color_dnn_result.png" ch="500" />

图18-16 可视化结果
## 2 实现几何图形分类

### 1 用前馈神经网络解决问题

我们下面要考验一下神经网络的能力。我们先用前面学过的全连接网络来解决这个问题，搭建一个三层的网络如下：

```Python
def dnn_model():
    num_output = 5
    max_epoch = 50
    batch_size = 16
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "pic_dnn")
    
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

样本数据为28x28的灰度图，所以我们要把它展开成1x784的向量，第一层用128个神经元，第二层用64个神经元，输出层5个神经元接Softmax分类函数。

最后可以得到如下训练结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_dnn_loss.png" />

图18-22 训练过程中损失函数值和准确度的变化

在测试集上得到的准确度是89.8%，这已经超出笔者的预期了，本来猜测准确度会小于80%。有兴趣的读者可以再精调一下这个前馈神经网络网络，看看是否可以得到更高的准确度。

### 2 用卷积神经网络解决问题

下面我们来看看卷积神经网络能不能完成这个工作。首先搭建网络模型如下：

```Python
def cnn_model():
    num_output = 5
    max_epoch = 50
    batch_size = 16
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "shape_cnn")
    
    c1 = ConvLayer((1,28,28), (8,3,3), (1,1), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    c2 = ConvLayer(p1.output_shape, (16,3,3), (1,0), params)
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
表18-2展示了模型中各层的作用和参数。

表18-2 模型各层的作用和参数

|ID|类型|参数|输入尺寸|输出尺寸|
|---|---|---|---|---|
|1|卷积|8x3x3, S=1,P=1|1x28x28|8x28x28|
|2|激活|Relu|8x28x28|8x28x28|
|3|池化|2x2, S=2, Max|8x28x28|8x14x14|
|4|卷积|16x3x3, S=1|8x14x14|16x12x12|
|5|激活|Relu|16x12x12|16x12x12|
|6|池化|2x2, S=2, Max|16x6x6|16x6x6|
|7|全连接|32|576|32|
|8|归一化||32|32|
|9|激活|Relu|32|32|
|10|全连接|5|32|5|
|11|分类|Softmax|5|5|

经过50个epoch的训练后，我们得到的结果如图18-23。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_cnn_loss.png" />

图18-23 训练过程中损失函数值和准确度的变化

以下是打印输出的最后几行：

```
......
epoch=49, total_iteration=14099
loss_train=0.002093, accuracy_train=1.000000
loss_valid=0.163053, accuracy_valid=0.944000
time used: 259.32207012176514
testing...
0.935
load parameters
0.96
```

可以看到我们在测试集上得到了96%的准确度，比前馈神经网络模型要高出很多，这也证明了卷积神经网络在图像识别上的能力。

图18-24是部分测试集中的测试样本的预测结果。

<img src='../Images/18/shape_cnn_result.png'/>

图18-24 测试结果

绝大部分样本预测是正确的，只有最后一个样本，看上去应该是一个很扁的三角形，被预测成了菱形。
## 3 解决MNIST分类问题

### 1 模型搭建

在12.1中，我们用一个三层的神经网络解决MNIST问题，并得到了97.49%的准确率。当时使用的模型如图18-31。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/nn3.png" ch="500" />

图18-31 前馈神经网络模型解决MNIST问题

这一节中，我们将学习如何使用卷积网络来解决MNIST问题。首先搭建模型如图18-32。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_net.png" />

图18-32 卷积神经网络模型解决MNIST问题

表18-5展示了模型中各层的功能和参数。

表18-5 模型中各层的功能和参数

|Layer|参数|输入|输出|参数个数|
|---|---|---|---|---|
|卷积层|8x5x5,s=1|1x28x28|8x24x24|200+8|
|激活层|2x2,s=2, max|8x24x24|8x24x24||
|池化层|Relu|8x24x24|8x12x12||
|卷积层|16x5x5,s=1|8x12x12|16x8x8|400+16|
|激活层|Relu|16x8x8|16x8x8||
|池化层|2x2, s=2, max|16x8x8|16x4x4||
|全连接层|256x32|256|32|8192+32|
|批归一化层||32|32||
|激活层|Relu|32|32||
|全连接层|32x10|32|10|320+10|
|分类层|softmax,10|10|10|

卷积核的大小如何选取呢？大部分卷积神经网络都会用1、3、5、7的方式递增，还要注意在做池化时，应该尽量让输入的矩阵尺寸是偶数，如果不是的话，应该在上一层卷积层加padding，使得卷积的输出结果矩阵的宽和高为偶数。

### 2 代码实现

```Python
def model():
    num_output = 10
    dataReader = LoadData(num_output)

    max_epoch = 5
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "mnist_conv_test")
    
    c1 = ConvLayer((1,28,28), (8,5,5), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 
  
    c2 = ConvLayer(p1.output_shape, (16,5,5), (1,0), params)
    net.add_layer(c2, "23")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")  

    f3 = FcLayer_2_0(p2.output_size, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BnLayer(f3.output_size)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")

    f4 = FcLayer_2_0(f3.output_size, 10, params)
    net.add_layer(f4, "f2")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    net.train(dataReader, checkpoint=0.05, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)
```

### 3 运行结果

训练5个epoch后的损失函数值和准确率的历史记录曲线如图18-33。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_loss.png" />

图18-33 训练过程中损失函数值和准确度的变化

打印输出结果如下：

```
...
epoch=4, total_iteration=2133
loss_train=0.054449, accuracy_train=0.984375
loss_valid=0.060550, accuracy_valid=0.982000
save parameters
time used: 513.3446323871613
testing...
0.9865
```

最后可以得到98.65%的准确率，比全连接网络要高1个百分点。如果想进一步提高准确率，可以尝试增加卷积层的能力，比如使用更多的卷积核来提取更多的特征。
## 4 Fashion-MNIST分类
### 1 用前馈神经网络来解决问题

#### 搭建模型

```Python
def dnn_model():
    num_output = 10
    max_epoch = 10
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "fashion_mnist_dnn")
    
    f1 = FcLayer_2_0(784, 128, params)
    net.add_layer(f1, "f1")
    bn1 = BnLayer(f1.output_size)
    net.add_layer(bn1, "bn1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")

    f2 = FcLayer_2_0(f1.output_size, 64, params)
    net.add_layer(f2, "f2")
    bn2 = BnLayer(f2.output_size)
    net.add_layer(bn2, "bn2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    
    f3 = FcLayer_2_0(f2.output_size, num_output, params)
    net.add_layer(f3, "f3")
    s3 = ClassificationLayer(Softmax())
    net.add_layer(s3, "s3")

    return net
```

#### 训练结果

训练10个epoch后得到如图18-37所示曲线，可以看到网络能力已经接近极限了，再训练下去会出现过拟合现象，准确度也不一定能提高。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistLoss_dnn.png" />

图18-37 训练过程中损失函数值和准确度的变化

图18-38是在测试集上的预测结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistResult_dnn.png" ch="555" />

图18-38 测试结果

凡是类别名字前面带*号的，表示预测错误，比如第3行第1列，本来应该是第7类“运动鞋”，却被预测成了“凉鞋”。

### 18.5.3 用卷积神经网络来解决问题

#### 搭建模型

```Python
def cnn_model():
    num_output = 10
    max_epoch = 10
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "fashion_mnist_conv_test")
    
    c1 = ConvLayer((1,28,28), (32,3,3), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    f3 = FcLayer_2_0(p1.output_size, 128, params)
    net.add_layer(f3, "f3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")

    f4 = FcLayer_2_0(f3.output_size, 10, params)
    net.add_layer(f4, "f4")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net
```

此模型只有一层卷积层，使用了32个卷积核，尺寸为3x3，后接最大池化层，然后两个全连接层。

#### 训练结果

训练10个epoch后得到如图18-39的曲线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistLoss_cnn.png" />

图18-39 训练过程中损失函数值和准确度的变化
### Python安装
![yang](./photo/1.png)
![yang](./photo/2.png)

![yang](./photo/3.png)
![yang](./photo/4.png)

### 学习心得
本次学习的卷积神经网络在神经网络技术中有着重要意义，这个技术的应用将使神经网络技术在AI里起到更多更全面的应用。卷积神经网络是深度学习中的一个里程碑式的技术，有了这个技术，才会让计算机有能力理解图片和视频信息，才会有计算机视觉的众多应用。这部分内容将会为我们逐步介绍卷积的前向计算、卷积的反向传播、池化的前向计算与反向传播，然后用代码实现一个卷积网络并训练一些实际数据。
