#  卷积神经网络

## 摘要

卷积神经网络是深度学习中的一个里程碑式的技术，有了这个技术，才会让计算机有能力理解图片和视频信息，才会有计算机视觉的众多应用。
# 第17章 卷积神经网络原理
卷积神经网络（Convolutional Neural Network, CNN）是一种 前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。它包括卷积层(alternating convolutional layer)和池层(pooling layer)。
### 能力
卷积神经网络是神经网络的类型之一，在图像识别和分类领域中取得了非常好的效果,为机器人、自动驾驶等应用提供了坚实的技术基础。
### 典型结构

一个典型的卷积神经网络的结构如图17-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_net.png" />

### 卷积核的作用
CNN中的卷积核
![](./picture/1.PNG)
图像处理时，给定输入图像，输入图像中一个小区域中像素加权平均后成为输出图像中的每个对应像素，其中权值由一个函数定义

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/circle_filters.png" ch="500" />

图17-6 卷积核的作用
##  前向计算
### 单入单出的二维卷积

二维卷积一般用于图像处理上。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_w3_s1.png" ch="526" />

图17-12 卷积运算的过程
###  卷积编程模型

图17-16侧重于解释五个概念的关系：

- 输入 Input Channel
- 卷积核组 WeightsBias
- 过滤器 Filter
- 卷积核 kernal
- 输出 Feature Map

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv3d.png" ch="500" />

图17-16 三通道经过两组过滤器的卷积过程
对于三维卷积，有以下特点：

1. 预先定义输出的feature map的数量，而不是根据前向计算自动计算出来
2. 对于每个输出，都有一个对应的过滤器Filter
3. 每个Filter内都有一个或多个卷积核Kernal，对应每个输入通道(Input Channel)
4. 每个Filter只有一个Bias值
5. 卷积核Kernal的大小一般是奇数

## 卷积前向计算代码实现

### 卷积核的实现

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


###  卷积前向运算的实现 - 方法1
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



我们试着运行上面的代码并循环10次，看看它的执行效率如何：

```
Time used for Python: 38.057225465774536
```

出乎我们的预料，在足足等了30多秒后，才返回结果。

通过试验发现，其运行速度非常慢，如果这样的函数在神经网络训练中被调用几十万次，其性能是非常糟糕的，这也是`Python`做为动态语言的一个缺点。

### 卷积前向运算的实现 - 方法2

我们把它编译成静态方法，是不是会快一些，[numba](https://numba.pydata.org/)，它可以在运行时把`Python`编译成`C`语言执行，代码是用`C`语言“风格”编写的`Python`代码，而且越像`C`的话，执行速度越快。

我们先用`pip`安装`numba`包：

```
pip install numba
```

然后在需要运行时编译的函数前面加上一个装饰符：

```Python
@nb.jit(nopython=True)
def jit_conv_4d(x, weights, bias, out_h, out_w, stride=1):
    ...
```

为了明确起见，我们把`conv_4d`前面加上一个`jit`前缀，表明这个函数是经过`numba`加速的。然后运行循环10次的测试代码：

```
Time used for Numba: 0.0727994441986084
```

这次只用了0.07秒，比纯`Python`代码快了500多倍

还需要检查一下其正确性。方法1输出结果为`output1`，Numba编译后的方法输出结果为`output2`，二者都是四维矩阵，我们用`np.allclose()`函数来比较它们的差异：

```Python
    print("correctness:", np.allclose(output1, output2, atol=1e-7))
```

得到的结果是：

```
correctness: True
```

`np.allclose`方法逐元素检查两种方法的返回值的差异，如果绝对误差在`1e-7`之内，说明两个返回的四维数组相似度极高，运算结果可信。

### 卷积前向运算的实现 - 方法3

由于卷积操作是原始图片数据与卷积核逐点相乘的结果，所以遍历每个点的运算速度非常慢。是否可以把卷积操作转换为矩阵操作呢？

在Caffe框架中，巧妙地把逐点相乘的运算转换成了矩阵运算，大大提升了程序运行速度。这就是著名的`im2col`函数（我们在代码中命名为`img2col`)。

```Python
    def forward_img2col(self, x, train=True):
        self.x = x
        self.batch_size = self.x.shape[0]
        assert(self.x.shape == (self.batch_size, self.InC, self.InH, self.InW))
        self.col_x = img2col(x, self.FH, self.FW, self.stride, self.padding)
        self.col_w = self.WB.W.reshape(self.OutC, -1).T
        self.col_b = self.WB.B.reshape(-1, self.OutC)
        out1 = np.dot(self.col_x, self.col_w) + self.col_b
        out2 = out1.reshape(batch_size, self.OutH, self.OutW, -1)
        self.z = np.transpose(out2, axes=(0, 3, 1, 2))
        return self.z
```

#### 原理

我们观察一下图17-20。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/img2col.png" ch="500" />

图17-20 把卷积运算转换成矩阵运算

先看上半部分：绿色的$3\times 3$矩阵为输入，经过棕色的卷积核运算后，得到右侧的$2\times 2$的矩阵。

再看图的下半部分：

第一步，上半部分中蓝色的虚线圆内的四个元素排列成第1行，形成[0,1,3,4]，红色虚线圆内的四个元素排列成第4行[4,5,7,8]，中间两行可以从右上角的[1,2,4,5]和左下角的[3,4,6,7]得到。这样，一个$3\times 3$的矩阵，就转换成了一个$4\times 4$的矩阵。也就是把卷积核视野中的每个二维$2\times 2$的数组变成$1\times 4$的向量。

第二步，把棕色的权重矩阵变成$4\times 1$的向量[3,2,1,0]。

第三步，把$4\times 4$的矩阵与$4\times 1$的向量相乘，得到$4\times 1$的结果向量[5,11,23,29]。

第四步：把$4\times 1$的结果变成$2\times 2$的矩阵，就得到了卷积运算的真实结果。

#### 权重数组的展开

对应的四维输入数据，卷积核权重数组也需要是四维的，其原始形状和展开后的形状如下：
```
weights=
(过滤器1)               (过滤器2)
    (卷积核1)               (卷积核1)
 [[[[ 0  1]             [[[12 13]
   [ 2  3]]               [14 15]]
    (卷积核2)               (卷积核2)
  [[ 4  5]               [[16 17]
   [ 6  7]]               [18 19]]
    (卷积核3)               (卷积核3)
  [[ 8  9]               [[20 21]
   [10 11]]]              [22 23]]]]
---------------------------------------
col_w=
 [[ 0 12]
  [ 1 13]
  [ 2 14]
  [ 3 15]
  [ 4 16]
  [ 5 17]
  [ 6 18]
  [ 7 19]
  [ 8 20]
  [ 9 21]
  [10 22]
  [11 23]]
```

至此，展开数组已经可以和权重数组做矩阵相乘了。

## 卷积层的训练
卷积层的训练需要从上一层回传的误差矩阵，然后计算：

1. 本层的权重矩阵的误差项
2. 本层的需要回传到下一层的误差矩阵

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_forward.png" />

图17-21 卷积正向运算
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_backward.png" />

图17-22 卷积运算中的误差反向传播
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/level3_1.png" ch="500" />

图17-28 原图和经过横边检测算子的卷积结果
## 卷积反向传播代码实现

### 方法1
把一些模块化的计算放到独立的函数中，用numba在运行时编译加速。

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
上面的代码中做了一些省略，只保留了基本的实现思路。
## 方法2

在前向计算中，我们试验了img2col的方法，取得了不错的效果。在反向传播中，也有对应的逆向方法，叫做col2img。下面我们基于它来实现另外一种反向传播算法，其基本思想是：把反向传播也看作是全连接层的方式，直接用矩阵运算代替卷积操作，然后把结果矩阵再转换成卷积操作的反向传播所需要的形状。

##  池化层
池化层设计的目的主要有两个。

最直接的目的，就是降低了下一层待处理的数据量。比如说，当卷积层的输出大小是32×32时，如果池化层过滤器的大小为2×2时，那么经过池化层处理后，输出数据的大小为16×16，也就是说现有的数据量一下子减少到池化前的1/4。当池化层最直接的目的达到了，那么它的间接目的也达到了：减少了参数数量，从而可以预防网络过拟合
![](./picture/2.PNG)
![](./picture/3.PNG)
### 17.5.1 常用池化方法

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

# 第18章 卷积神经网络应用
## 目标检测
图像识别中，目标检测的任务，是对输入图像样本准确进行分类的基础上，检测其中包含的某些目标，并对它们准确定位并标识。
##　目标定位
图像分类问题一般都采用Softmax回归来解决，最后输出的结果是一个多维列向量，且向量的维数与假定的分类类别数一致。在此基础上希望检测其中的包含的各种目标并对它们进行定位，这里对这个监督学习任务的标签表示形式作出定义。
![](./picture/4.PNG)
##　滑窗检测
用以实现目标检测的算法之一叫做滑窗检测（Sliding Windows Detection）。
滑动窗口检测算法的实现，首先需要用卷积网络训练出一个能够准确识别目标的分类器，且这个分类器要尽可能采用仅包含该目标的训练样本进行训练。随后选定一个特定大小（小于输入图像大小）的窗口，在要检测目标的样本图像中以固定的步幅滑动这个窗口，从上到下，从左到右依次遍历整张图像，同时把窗口经过的区域依次输入之前训练好的分类器中，以此实现对目标及其位置的检测。
![](./picture/5.PNG)

##  实现颜色分类
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/color_sample.png" ch="500" />

图18-14 颜色分类样本数据

### 用前馈神经网络解决问题

#### 数据处理

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

向量[0.299,0.587,0.114]的作用是，把三通道的彩色图片的RGB值与此向量相乘，得到灰度图，三个因子相加等于1，这样如果原来是[255,255,255]的话，最后的灰度图的值还是255。

#### 搭建模型

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

### 用卷积神经网络解决问题
我们直接使用三通道的彩色图片，不需要再做数据转换了。

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

#### 运行结果

经过20个epoch的训练后，得到的结果如图18-17。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/color_cnn_loss.png" />

图18-17 训练过程中的损失函数值和准确度变化曲线

图18-18是测试集中前64个测试样本的预测结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/color_cnn_result.png" ch="500" />

图18-18 测试结果

###  1x1卷积
我们以三通道的数据举例。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/OneByOne.png" ch="500" />

图18-19 1x1卷积核的工作原理

假设有一个三通道的1x1的卷积核，其值为[1,2,-1]，则相当于把每个通道的同一位置的像素值乘以卷积核，然后把结果相加，作为输出通道的同一位置的像素值。以左上角的像素点为例：

$$
1 \times 1 + 1 \times 2 + 1 \times (-1)=2
$$

相当于把上图拆开成9个样本，其值为：

```
[1,1,1] # 左上角点
[3,3,0] # 中上点
[0,0,0] # 右上角点
[2,0,0] # 左中点
[0,1,1] # 中点
[4,2,1] # 右中点
[1,1,1] # 左下角点
[2,1,1] # 下中点
[0,0,0] # 右下角点
```

上述值排成一个9行3列的矩阵，然后与一个3行1列的向量$(1,2,-1)^T$相乘，得到9行1列的向量，然后再转换成3x3的矩阵。当然在实际过程中，这个1x1的卷积核的数值是学习出来的，而不是人为指定的。

这样做可以达到两个目的：

1. 跨通道信息整合
2. 降维以减少学习参数

所以1x1的卷积核关注的是不同通道的相同位置的像素之间的相关性，而不是同一通道内的像素的相关性，在本例中，意味着它关心的彩色通道信息，通过不同的卷积核，把彩色通道信息转变成另外一种表达方式，在保留原始信息的同时，还实现了降维。
### 颜色分类可视化解释

在这里笔者根据自己的理解，解释一下针对这个颜色分类问题，卷积神经网络是如何工作的。
<img src='../Images/18/color_cnn_visualization.png'/>

图18-20 颜色分类问题的可视化解释
## 实现几何图形分类
几何图形可总分为：立体几何图形和平面几何图形；其细分如下：

一、立体几何图形可以分为以下几类：

1、柱体：包括圆柱和棱柱。棱柱又可分为直棱柱和斜棱柱，按底面边数的多少又可分为三棱柱、四棱柱、N棱柱；

2、锥体：包括圆锥体和棱锥体，棱锥分为三棱锥、四棱锥及N棱锥；棱锥体积为 ;

3、旋转体：包括圆柱、圆台、圆锥、球、球冠、弓环、圆环、堤环、扇环、枣核形等；

4、截面体：包括棱台、圆台、斜截圆柱、斜截棱柱、斜截圆锥、球冠、球缺等。

### 用前馈神经网络解决问题

用前面学过的全连接网络来解决这个问题，搭建一个三层的网络如下：

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
训练结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_dnn_loss.png" />

图18-22 训练过程中损失函数值和准确度的变化

### 用卷积神经网络解决问题
首先搭建网络模型如下：

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


<img src='../Images/18/shape_cnn_result.png'/>

图18-24 测试结果

### 形状分类可视化解释

<img src='../Images/18/shape_cnn_visualization.png'/>

图18-25 可视化解释

参看图18-25，表18-3解释了8个卷积核的作用。

表18-3 8个卷积核的作用

|卷积核序号|作用|直线|三角形|菱形|矩形|圆形|
|:--:|---|:--:|:--:|:--:|:--:|:--:|
|1|左侧边缘|0|1|0|1|1|
|2|大色块区域|0|1|1|1|1|
|3|左上侧边缘|0|1|1|0|1|
|4|45度短边|1|1|1|0|1|
|5|右侧边缘、上横边|0|0|0|1|1|
|6|左上、右上、右下|0|1|1|0|1|
|7|左边框和右下角|0|0|0|1|1|
|8|左上和右下，及背景|0|0|1|0|1|

表18-3中，左侧为卷积核的作用，右侧为某个特征对于5种形状的判别力度，0表示该特征无法找到，1表示可以找到该特征。

1. 比如第一个卷积核，其作用为判断是否有左侧边缘，那么第一行的数据为[0,1,0,1,1]，表示对直线和菱形来说，没有左侧边缘特征，而对于三角形、矩形、圆形来说，有左侧边缘特征。这样的话，就可以根据这个特征把5种形状分为两类：

   - A类有左侧边缘特征：三角形、矩形、圆形
   - B类无左侧边缘特征：直线、菱形

2. 再看第二个卷积核，是判断是否有大色块区域的，只有直线没有该特征，其它4种形状都有。那么看第1个特征的B类种，包括直线、菱形，则第2个特征就可以把直线和菱形分开了。

3. 然后我们只关注A类形状，看第三个卷积核，判断是否有左上侧边缘，对于三角形、矩形、圆形的取值为[1,0,1]，即矩形没有左上侧边缘，这样就可以把矩形从A类中分出来。

4. 对于三角形和圆形，卷积核5、7、8都可以给出不同的值，这就可以把二者分开了。
## 实现几何图形及颜色分类
### 用前馈神经网络解决问题

我们仍然先使用全连接网络来解决这个问题，搭建一个三层的网络如下：

```Python
ef dnn_model():
    num_output = 9
    max_epoch = 50
    batch_size = 16
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "color_shape_dnn")
    
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

训练50个epoch后可以得到如下如图18-27所示的训练结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_color_dnn_loss.png" />

图18-27 训练过程中损失函数值和准确度的变化

```
......
epoch=49, total_iteration=15199
loss_train=0.003370, accuracy_train=1.000000
loss_valid=0.510589, accuracy_valid=0.883333
time used: 25.34346342086792
testing...
0.9011111111111111
load parameters
0.8988888888888888
```

在测试集上得到的准确度是89%，这已经超出笔者的预期了，本来猜测准确度会小于80%。有兴趣的读者可以再精调一下这个前馈神经网络网络，看看是否可以得到更高的准确度。

图18-28是部分测试集中的测试样本的预测结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_color_dnn_result.png" ch="500" />

图18-28 测试结果

绝大部分样本预测是正确的，但是第3行第2列的样本，应该是green-rect，被预测成green-circle；最后两行的两个green-tri也被预测错了形状，颜色并没有错。

### 用卷积神经网络解决问题

下面我们来看看卷积神经网络能不能完成这个工作。首先搭建网络模型如下：

```Python
def cnn_model():
    num_output = 9
    max_epoch = 20
    batch_size = 16
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "shape_color_cnn")
    
    c1 = ConvLayer((3,28,28), (8,3,3), (1,1), params)
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

经过20个epoch的训练后，我们得到的结果如图18-29。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_color_cnn_loss.png" />

图18-29 训练过程中损失函数值和准确度的变化

以下是打印输出的最后几行：

```
......
epoch=19, total_iteration=6079
loss_train=0.005184, accuracy_train=1.000000
loss_valid=0.118708, accuracy_valid=0.957407
time used: 131.77996039390564
testing...
0.97
load parameters
0.97
```

可以看到我们在测试集上得到了97%的准确度，比DNN模型要高出很多，这也证明了卷积神经网络在图像识别上的能力。

图18-30是部分测试集中的测试样本的预测结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_color_cnn_result.png" ch="500" />

图18-30 测试结果

绝大部分样本预测是正确的，只有最后一行第4个样本，本来是green-triangle，被预测成green-circle。

##  解决MNIST分类问题

### 模型搭建
首先搭建模型如图18-32。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_net.png" />

图18-32 卷积神经网络模型解决MNIST问题

### 18.4.2 代码实现

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

### 18.4.3 运行结果

训练5个epoch后的损失函数值和准确率的历史记录曲线如图18-33。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_loss.png" />

图18-33 训练过程中损失函数值和准确度的变化
### 可视化
#### 第一组的卷积可视化

下图按行显示了以下内容：

1. 卷积核数值
2. 卷积核抽象
3. 卷积结果
4. 激活结果
5. 池化结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_layer_123_filter.png" ch="500" />

图18-34 卷积结果可视化

卷积核是5x5的，一共8个卷积核，所以第一行直接展示了卷积核的数值图形化以后的结果，但是由于色块太大，不容易看清楚其具体的模式，那么第二行的模式是如何抽象出来的呢？

因为特征是未知的，所以卷积神经网络不可能学习出类似下面的两个矩阵中左侧矩阵的整齐的数值，而很可能是如同右侧的矩阵一样具有很多噪音，但是大致轮廓还是个左上到右下的三角形，只是一些局部点上有一些值的波动。

```
2  2  1  1  0               2  0  1  1  0
2  1  1  0  0               2  1  1  2  0
1  1  0 -1 -2               0  1  0 -1 -2
1  0 -1 -2 -3               1 -1  1 -4 -3
0 -1 -2 -3 -4               0 -1 -2 -3 -2
```

如何“看”出一个大概符合某个规律的模板呢？对此，笔者的心得是：

1. 摘掉眼镜（或者眯起眼睛）看第一行的卷积核的明暗变化模式；
2. 也可以用图像处理的办法，把卷积核形成的5x5的点阵做一个模糊处理；
3. 结合第三行的卷积结果推想卷积核的行为。

#### 第二组的卷积可视化

图18-35是第二组的卷积、激活、池化层的输出结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_layer_456.png" ch="500" />

图18-35 第二组卷积核、激活、池化的可视化

- Conv2：由于是在第一层的特征图上卷积后叠加的结果，所以基本不能按照原图理解，但也能大致看出是是一些轮廓抽取的作用；
- Relu2：能看出的是如果黑色区域多的话，说明基本没有激活值，此卷积核效果就没用；
- Pool2：池化后分化明显的特征图是比较有用的特征，比如3、6、12、15、16；信息太多或者太少的特征图，都用途偏小，比如1、7、10、11。
# Fashion-MNIST分类
### 提出问题

MNIST手写识别数据集，对卷积神经网络来说已经太简单了，于是科学家们增加了图片的复杂度，用10种物品代替了10个数字，图18-36是它们的部分样本。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistSample.png" ch="500" />

图18-36 部分样本图展示

每3行是一类样本，按样本类别（从0开始计数）分行显示：

0. T-Shirt，T恤衫（1-3行）
1. Trouser，裤子（4-6行）
2. Pullover，套头衫（7-9行）
3. Dress，连衣裙（10-12行）
4. Coat，外套（13-15行）
5. Sandal，凉鞋（16-18行）
6. Shirt，衬衫（19-21行）
7. Sneaker，运动鞋（22-24行）
8. Bag，包（25-27行）
9. Ankle Boot，短靴（28-30行）

### 用前馈神经网络来解决问题

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
### 用卷积神经网络来解决问题

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
训练10个epoch后得到如图18-39的曲线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistLoss_cnn.png" />

图18-39 训练过程中损失函数值和准确度的变化

在测试集上得到91.12%的准确率，在测试集上的前几个样本的预测结果如图18-40所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistResult_cnn.png" ch="555" />

图18-40 测试结果
## Cifar-10分类
Cifar 是加拿大政府牵头投资的一个先进科学项目研究所。Hinton、Bengio和他的学生在2004年拿到了 Cifar 投资的少量资金，建立了神经计算和自适应感知项目。这个项目结集了不少计算机科学家、生物学家、电气工程师、神经科学家、物理学家、心理学家，加速推动了 Deep Learning 的进程。从这个阵容来看，DL 已经和 ML 系的数据挖掘分的很远了。Deep Learning 强调的是自适应感知和人工智能，是计算机与神经科学交叉；Data Mining 强调的是高速、大数据、统计数学分析，是计算机和数学的交叉。

Cifar-10 是由 Hinton 的学生 Alex Krizhevsky、Ilya Sutskever 收集的一个用于普适物体识别的数据集。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/cifar10_sample.png" ch="500" />

图18-41 Cifar-10样本数据
### 环境搭建
我们将使用Keras$^{[1]}$来训练模型，因为Keras是一个在TensorFlow平台上经过抽象的工具，它的抽象思想与我们在前面学习过的各种Layer的概念完全一致，有利于读者在前面的基础上轻松地继续学习。环境搭建有很多细节，我们在这里不详细介绍，只是把基本步骤列出。

1. 安装Python 3.6（本书中所有案例在Python 3.6上开发测试）
2. 安装CUDA（没有GPU的读者请跳过）
3. 安装cuDNN（没有GPU的读者请跳过）
4. 安装TensorFlow，有GPU硬件的一定要按照GPU版，没有的只能安装CPU版
5. 安装Keras

安装好后用pip list看一下，关键的几个包是：

```
Package              Version
-------------------- ---------
Keras                2.2.5
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.0
matplotlib           3.1.1
numpy                1.17.0
tensorboard          1.13.1
tensorflow-estimator 1.13.0
tensorflow-gpu       1.13.1
```

如果没有GPU，则"tensorflow-gpu"一项会是"tensorflow"。
### 代码实现

```Python
batch_size = 32
num_classes = 10
epochs = 25
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    ...
```

在这个模型中：

1. 先用卷积->激活->卷积->激活->池化->丢弃层，做为第一梯队，卷积核32个；
2. 然后再用卷积->激活->卷积->激活->池化->丢弃层做为第二梯队，卷积核64个；
3. Flatten和Dense相当于把池化的结果转成Nx512的全连接层，N是池化输出的尺寸，被Flatten扁平化了；
4. 再接丢弃层，避免过拟合；
5. 最后接10个神经元的全连接层加Softmax输出。

为什么每一个梯队都要接一个DropOut层呢？因为这个网络结果设计已经比较复杂了，对于这个问题来说很可能会过拟合，所以要避免过拟合。如果简化网络结构，又可能会造成训练时间过长而不收敛。
### 训练结果

#### 在GPU上训练

在GPU上训练，每一个epoch大约需要1分钟；而在一个8核的CPU上训练，每个epoch大约需要2分钟（据笔者观察是因为并行计算占满了8个核）。所以即使读者没有GPU，用CPU训练还是可以接受的。以下是在GPU上的训练输出：

```
Epoch 1/25
1563/1563 [==============================] - 33s 21ms/step - loss: 1.8770 - acc: 0.3103 - val_loss: 1.6447 - val_acc: 0.4098
......
Epoch 25/25
1563/1563 [==============================] - 87s 55ms/step - loss: 0.8809 - acc: 0.6960 - val_loss: 0.7724 - val_acc: 0.7372

Test loss: 0.772429921245575
Test accuracy: 0.7372
```
经过25轮后，模型在测试集上的准确度为73.72%。

#### 在CPU上训练

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

