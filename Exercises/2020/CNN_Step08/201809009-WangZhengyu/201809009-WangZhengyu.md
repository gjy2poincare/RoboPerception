# 第4次作业

## 作业要求

学习Step08-CNN的内容：主要记录笔记，特别是对理论的掌握和理解；

主体内容上要求包含：

（1）软件安装，

（2）代码测试、分析和理解（可以在源代码中用“//”注释的方式来展示分析和理解过程），

（3）结果分析和总结，要求有心得体会

（4）代码修改

要求图文并茂，图片应尽量与书本有所区别；

完成时间：下一次课程上课前。

## 学号：201809009   姓名：王征宇

### 0、软件安装

前提：首先在网址上下载python3.6.5

1、打开E:\rengongzhineng\ai-edu\A-基础教程\A2-神经网络基本原理简明教程\SourceCode\ch17-CNNBasic路径下的Level0_KnowCNN.py代码
2、运行代码  出现错误发现找不到组件

![](./Images/20.png)

3、安装组件opencv

```python
pip install opencv-python
``` 

![](./Images/21.png)


4、运行代码  出现错误发现找不到组件 matplotlib
5、安装组件matplotlib

```
pip install matplotlib
```

![](./Images/22.png)

6、再次运行 运行成功 软件配置完成

![](./Images/23.png)

### 一、CH08 卷积神经网络

#### 1、卷积神经网络

#### （1）卷积网络的典型结构

一个典型的卷积神经网络的结构如下图所示：

![](./Images/conv_net.png)

在一个典型的卷积神经网络中，会至少包含以下几个层：
- 卷积层
- 激活函数层
- 池化层
- 全连接分类层
   
#### (2)卷积核的作用
卷积核其实就是一个小矩阵，类似这样：

```
1.1  0.23  -0.45
0.1  -2.1   1.24
0.74 -1.32  0.01
```

这是一个3x3的卷积核，还会有1x1、5x5、7x7、9x9、11x11的卷积核。在卷积层中，我们会用输入数据与卷积核相乘，得到输出数据，就类似全连接层中的Weights一样，所以卷积核里的数值，也是通过反向传播的方法学习到的。

卷积核的具体作用：

![](./Images/circle_filters.png)

上面的九张图，是使用9个不同的卷积核在同一张图上运算后得到的结果，而下面的表格中按顺序列出了9个卷积核的数值和名称，可以一一对应到上面的9张图中：

||1|2|3|
|---|---|---|---|
|1|0,-1, 0<br>-1, 5,-1<br>0,-1, 0|0, 0, 0 <br> -1, 2,-1 <br> 0, 0, 0|1, 1, 1 <br> 1,-9, 1 <br> 1, 1, 1|
||sharpness|vertical edge|surround|
|2|-1,-2, -1 <br> 0, 0, 0<br>1, 2, 1|0, 0, 0 <br> 0, 1, 0 <br> 0, 0, 0|0,-1, 0 <br> 0, 2, 0 <br> 0,-1, 0|
||sobel y|nothing|horizontal edge|
|3|0.11,0.11,0.11 <br>0.11,0.11,0.11<br>0.11,0.11,0.11|-1, 0, 1 <br> -2, 0, 2 <br> -1, 0, 1|2, 0, 0 <br> 0,-1, 0 <br> 0, 0,-1|
||blur|sobel x|embossing|
9个卷积核的作用：

|序号|名称|说明|
|---|---|---|
|1|锐化|如果一个像素点比周围像素点亮，则此算子会令其更亮|
|2|检测竖边|检测出了十字线中的竖线，由于是左侧和右侧分别检查一次，所以得到两条颜色不一样的竖线|
|3|周边|把周边增强，把同色的区域变弱，形成大色块|
|4|Sobel-Y|纵向亮度差分可以检测出横边，与横边检测不同的是，它可以使得两条横线具有相同的颜色，具有分割线的效果|
|5|Identity|中心为1四周为0的过滤器，卷积后与原图相同|
|6|横边检测|检测出了十字线中的横线，由于是上侧和下侧分别检查一次，所以得到两条颜色不一样的横线|
|7|模糊|通过把周围的点做平均值计算而"杀富济贫"造成模糊效果|
|8|Sobel-X|横向亮度差分可以检测出竖边，与竖边检测不同的是，它可以使得两条竖线具有相同的颜色，具有分割线的效果|
|9|浮雕|形成大理石浮雕般的效果|
#### (3)卷积后续的运算
下图中的四个子图，依次展示了：
1. 原图
2. 卷积结果
3. 激活结果
4. 池化结果

![](./Images/circle_conv_relu_pool.png)

1. 注意图一是原始图片，用cv2读取出来的图片，其顺序是反向的，即：
- 第一维是高度
- 第二维是宽度
- 第三维是彩色通道数，但是其顺序为BGR，而不是常用的RGB
2. 我们对原始图片使用了一个3x1x3x3的卷积核，因为原始图片为彩色图片，所以第一个维度是3，对应RGB三个彩色通道；我们希望只输出一张feature map，以便于说明，所以第二维是1；我们使用了3x3的卷积核，用的是sobel x算子。所以图二是卷积后的结果。
3. 图三做了一层Relu激活计算，把小于0的值都去掉了，只留下了一些边的特征。
4. 图四是图三的四分之一大小，虽然图片缩小了，但是特征都没有丢失，反而因为图像尺寸变小而变得密集，亮点的密度要比图三大而粗。

### 二、卷积的前向计算
#### 1、卷积的前向计算原理
#### （1）单入多出的升维卷积

原始输入是一维的图片，但是我们可以用多个卷积核分别对其计算，从而得到多个特征输出。如下图所示：

![](./Images/conv_2w3.png)

一张4x4的图片，用两个卷积核并行地处理，输出为2个2x2的图片。在训练过程中，这两个卷积核会完成不同的特征学习。

#### （2）多入单出的降维卷积

一张图片，通常是彩色的，具有红绿蓝三个通道。我们可以有两个选择来处理：

![](./Images/weights3d.png)

1. 变成灰度的，每个像素只剩下一个值，就可以用二维卷积
2. 对于三个通道，每个通道都使用一个卷积核，分别处理红绿蓝三种颜色的信息

显然第2种方法可以从图中学习到更多的特征，于是出现了三维卷积，即有三个卷积核分别对应书的三个通道，三个子核的尺寸是一样的，比如都是2x2，这样的话，这三个卷积核就是一个3x2x2的立体核，称为过滤器Filter，所以称为三维卷积。

![](./Images/multiple_filter.png)

在上图中，每一个卷积核对应着左侧相同颜色的输入通道，三个过滤器的值并不一定相同。对三个通道各自做卷积后，得到右侧的三张特征图，然后在按照原始值不加权地相加在一起，得到最右侧的黑色特征图，这张图里面已经把三种颜色的特征混在一起了，所以画成了黑色。

虽然输入图片是多个通道的，或者说是三维的，但是在相同数量的过滤器的计算后，相加在一起的结果是一个通道，即2维数据，所以称为降维。这当然简化了对多通道数据的计算难度，但同时也会损失多通道数据自带的颜色信息。

#### （3）多入多出的同维卷积

在上面的例子中，是一个过滤器Filter内含三个卷积核Kernal。我们假设有一个彩色图片为3x3的，如果有两组3x2x2的卷积核的话，会做什么样的卷积计算？看下图：

![](./Images/conv3dp.png)
第一个过滤器Filter-1为棕色所示，它有三卷积核(Kernal)，命名为Kernal-1，Keanrl-2，Kernal-3，分别在红绿蓝三个输入通道上进行卷积操作，生成三个2x2的输出Feature-1,n。然后三个Feature-1,n相加，并再加上b1偏移值，形成最后的棕色输出Result-1。

对于灰色的过滤器Filter-2也是一样，先生成三个Feature-2,n，然后相加再加b2，最后得到Result-2。

之所以Feature-m,n还用红绿蓝三色表示，是因为在此时，它们还保留着红绿蓝三种色彩的各自的信息，一旦相加后得到Result，这种信息就丢失了。
#### （4）步长 stride

前面的例子中，每次计算后，卷积核会向右或者向下移动一个单元，即步长stride = 1。而在下面这个卷积操作中，卷积核每次向右或向下移动两个单元，即stride = 2。

![](./Images/Stride2.png)

在后续的步骤中，由于每次移动两格，所以最终得到一个2x2的图片。

#### （5）填充 padding

如果原始图为4x4，用3x3的卷积核进行卷积后，目标图片变成了2x2。如果我们想保持目标图片和原始图片为同样大小，该怎么办呢？一般我们会向原始图片周围填充一圈0，然后再做卷积。如下图：

![](./Images/padding.png)

#### （6） 输出结果

综合以上所有情况，可以得到卷积后的输出图片的大小的公式：

$$
H_{Output}= {H_{Input} - H_{Kernal} + 2Padding \over Stride} + 1
$$

$$
W_{Output}= {W_{Input} - W_{Kernal} + 2Padding \over Stride} + 1
$$
###  三、卷积前向计算代码实现
#### 1、卷积核的实现

```Python
class ConvWeightsBias(WeightsBias_2_1):
    def __init__(self, output_c, input_c, filter_h, filter_w, init_method, optimizer_name, eta):
        self.FilterCount = output_c:# 一维：滤波
        self.KernalCount = input_c:# 二维：内核
        self.KernalHeight = filter_h:# 三维：高度
        self.KernalWidth = filter_w:# 四维：宽度
        ...

    def Initialize(self, folder, name, create_new)::# 初始化
        self.WBShape = (self.FilterCount, self.KernalCount, self.KernalHeight, self.KernalWidth)        
        ...

    def CreateNew(self)::# 创建一个新的
        self.W = ConvWeightsBias.InitialConvParameters(self.WBShape, self.init_method)
        self.B = np.zeros((self.FilterCount, 1))
...
```
![](./Images/ConvWeightsBias.png)

#### 2、 前向运算实现的3种方法
 - 方法1
  ```Python
class ConvLayer(CLayer):
    def forward(self, x, train=True):# 前向传递
        self.x = x
        self.batch_size = self.x.shape[0]
        # 如果有必要的话，先对输入矩阵做padding
        if self.padding > 0:
            self.padded = np.pad(...)# 步长
        else:
            self.padded = self.x# 步长
        #end if
        self.z = conv_4d(...)# 四维
        return self.z
```
- 方法2
```Python
    # 演习
    output2 = jit_conv_4d(x, wb.W, wb.B, output_height, output_width, stride)
    # 运行
    s2 = time.time()
    for i in range(10):#循环输出conv_4d
        output2 = jit_conv_4d(x, wb.W, wb.B, output_height, output_width, stride)
    e2 = time.time()
    print("Time used for Numba:", e2 - s2)# 声明
```
- 方法3

```Python
    def forward_img2col(self, x, train=True)::# 前向传递
        self.x = x
        self.batch_size = self.x.shape[0]# 批大小
        assert(self.x.shape == (self.batch_size, self.InC, self.InH, self.InW))
        self.col_x = img2col(x, self.FH, self.FW, self.stride, self.padding)
        self.col_w = self.WB.W.reshape(self.OutC, -1).T# w
        self.col_b = self.WB.B.reshape(-1, self.OutC)# b 
        out1 = np.dot(self.col_x, self.col_w) + self.col_b# 矩阵相乘
        out2 = out1.reshape(batch_size, self.OutH, self.OutW, -1)
        self.z = np.transpose(out2, axes=(0, 3, 1, 2))
        return self.z
```

#### 3、性能测试


下面我们比较一下方法2和方法3的性能。

```Python
def test_performance():
    ...
    print("compare correctness of method 1 and method 2:")
    print("forward:", np.allclose(f1, f2, atol=1e-7))
```

上述代码先生成一个`ConvLayer`实例，然后分别调用1000次`forward_numba()`方法和1000次`forward_img2col()`方法，最后得到的结果是：

```
method numba: 11.663846492767334
method img2col: 14.926148653030396
compare correctness of method 1 and method 2:
forward: True
```
![](./Images/6.png)

![](./Images/7.png)
`numba`方法会比`im2col`方法快3秒，目前看来`numba`方法稍占优势。但是如果没有`numba`的帮助，`im2col`方法会比方法1快几百倍。

### 三、卷积的反向传播原理

#### 1、计算反向传播的梯度矩阵

正向公式：

$$Z = W*A+b \tag{0}$$

其中，W是卷积核，*表示卷积（互相关）计算，A为当前层的输入项，b是偏移（未在图中画出），Z为当前层的输出项，但尚未经过激活函数处理。
#### 2、有多个卷积核时的梯度计算

有多个卷积核也就意味着有多个输出通道。

![](./Images/conv_2w2.png)

#### 3、有多个输入时的梯度计算

当输入层是多个图层时，每个图层必须对应一个卷积核，如下图：

![](./Images/conv_1W222.png)

#### 4、权重（卷积核）梯度计算

![](./Images/conv_forward.png)

要求J对w11的梯度，从正向公式可以看到，w11对所有的z都有贡献，所以：

$$
\frac{\partial J}{\partial w_{11}} = \frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial w_{11}} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial w_{11}} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial w_{11}} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial w_{11}}
$$
$$
=\delta_{z11} \cdot a_{11} + \delta_{z12} \cdot a_{12} + \delta_{z21} \cdot a_{21} + \delta_{z22} \cdot a_{22} \tag{9}
$$

对W22也是一样的：

$$
\frac{\partial J}{\partial w_{12}} = \frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial w_{12}} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial w_{12}} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial w_{12}} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial w_{12}}
$$
$$
=\delta_{z11} \cdot a_{12} + \delta_{z12} \cdot a_{13} + \delta_{z21} \cdot a_{22} + \delta_{z22} \cdot a_{23} \tag{10}
$$
#### 5、计算卷积核梯度的实例
训练部分的代码实现如下：

```Python
def train(x, w, b, y):
    output = create_zero_array(x, w)
    for i in range(10000):
        # forward
        jit_conv_2d(x, w, b, output)
        # loss
        t1 = (output - y)
        m = t1.shape[0]*t1.shape[1]
        LOSS = np.multiply(t1, t1)
        loss = np.sum(LOSS)/2/m
        print(i,loss)
        if loss < 1e-7:
            break
        # delta
        delta = output - y
        # backward
        dw = np.zeros(w.shape)
        jit_conv_2d(x, delta, b, dw)
        w = w - 0.5 * dw/m
    #end for
    return w
```

一共迭代10000次：
1. 用jit_conv_2d(x,w...)做一次前向计算
2. 计算loss值以便检测停止条件，当loss值小于1e-7时停止迭代
3. 然后计算delta值
4. 再用jit_conv_2d(x,delta)做一次反向计算，得到w的梯度
5. 最后更新卷积核w的值

主过程如下：

```Python
if __name__ == '__main__':
    # 创建样本数据
    x, w_true, y = create_sample_image()
    # 随机初始化卷积核
    w_init = np.random.normal(0, 0.1, w_true.shape)
    # 训练
    w_result = train(x,w_init,0,y)
    # 打印比较真实卷积核值和训练出来的卷积核值
    print("w_true:\n", w_true)
    print("w_result:\n", w_result)
    # 用训练出来的卷积核值对原始图片进行卷积
    y_hat = np.zeros(y.shape)
    jit_conv_2d(x, w_true, 0, y_hat)
    # 与真实的卷积核的卷积结果比较
    show_result(x, w_true, w_result)
    # 比较卷积核值的差异核卷积结果的差异
    print("w allclose:", np.allclose(w_true, w_result, atol=1e-2))
    print("y allclose:", np.allclose(y, y_hat, atol=1e-2))
```

运行结果：

```
......
3458 1.0063169744079507e-07
3459 1.0031151142628902e-07
3460 9.999234418532805e-08
w_true:
 [[ 0 -1  0]
 [ 0  2  0]
 [ 0 -1  0]]
w_result:
 [[-1.86879237e-03 -9.97261724e-01 -1.01212359e-03]
 [ 2.58961697e-03  1.99494606e+00  2.74435794e-03]
 [-8.67754199e-04 -9.97404263e-01 -1.87580756e-03]]
w allclose: True
y allclose: True
```
### 四、卷积反向传播代码实现
- 方法1
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
- 方法2
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


### 3、正确性与性能测试

在正向计算中，numba稍胜一筹，下面我们来测试一下二者的反向计算性能，然后比较梯度输出矩阵的结果来验证正确性。

```Python
def test_performance():
    ...
```

先用numba方法测试1000次的正向+反向，然后再测试1000次img2col的正向+反向，同时我们会比较反向传播的三个输出值：误差矩阵b、权重矩阵梯度dw、偏移矩阵梯度db。

输出结果：

```
method numba: 11.830008506774902
method img2col: 3.543151378631592
compare correctness of method 1 and method 2:
forward: True
backward: True
dW: True
dB: True
```

![](./Images/8.png)


![](./Images/9.png)
这次img2col方法完胜，100次的正向+反向用了3.5秒，而numba方法花费的时间是前者的三倍多。再看二者的前向计算和反向传播的结果比较，四个矩阵的比较全都是True，说明我们的代码是没有问题的。

那么我们能不能混用numba方法的前向计算和img2col方法的反向传播呢？不行。因为img2col方法的反向传播需要用到其正向计算方法中的两个缓存数组，一个是输入数据的矩阵变换结果，另一个是权重矩阵变换结果，所以img2col方法必须正向反向配合使用。


## 五、池化层

### 1、常用池化方法

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

### 2、 实现方法1

按照标准公式来实现池化的正向和反向代码。

```Python
class PoolingLayer(CLayer):
    def forward_numba(self, x, train=True):
        ......

    def backward_numba(self, delta_in, layer_idx):
        ......
```

有了前面的经验，这次我们直接把前向和反向函数用numba方式来实现，并在前面加上@nb.jit修饰符：
```Python
@nb.jit(nopython=True)
def jit_maxpool_forward(...):
    ...
    return z

@nb.jit(nopython=True)
def jit_maxpool_backward(...):
    ...
    return delta_out
```

### 3、 实现方法2

池化也有类似与卷积优化的方法来计算，在图17-36中，我们假设大写字母为池子中的最大元素，并且用max_pool方式。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/img2col_pool.png" />

图17-36 池化层的img2col实现

原始数据先做img2col变换，然后做一次np.max(axis=1)的max计算，会大大增加速度，然后把结果reshape成正确的矩阵即可。做一次大矩阵的max计算，比做4次小矩阵计算要快很多。

```Python
class PoolingLayer(CLayer):
    def forward_img2col(self, x, train=True):
        ......

    def backward_col2img(self, delta_in, layer_idx):
        ......
```

### 5、性能测试

下面我们要比较一下以上两种实现方式的性能，来最终决定使用哪一种。

对同样的一批64个样本，分别用两种方法做5000次的前向和反向计算，得到的结果：

```
Elapsed of numba: 20
Elapsed of img2col: 46
forward: True
backward: True
```

numba方法用了20秒，img2col方法用了46秒。并且两种方法的返回矩阵值是一样的，说明代码实现正确。

![](./Images/1.png)

## 六、经典卷积神经网络模型

###  1、AlexNet (2012)
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


### 2、ZFNet (2013)

图18-3是ZFNet的结构示意图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ZFNet.png" />

### 3、VGGNet (2015)


图18-7为VGG16（16层的VGG）模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_Vgg16.png" ch="500" />


### 4、GoogLeNet (2014)

图18-8是GoogLeNet的模型结构图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_GoogleNet.png" />

图18-8 GoogLeNet模型结构图

蓝色为卷积运算，红色为池化运算，黄色为softmax分类。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_GoogleNet_Inception.png" />

### 5、 ResNets (2015)

图18-10是ResNets的模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ResNet.png" />

图18-10 ResNets模型结构图

## 七、实现颜色分类


### 1、用前馈神经网络解决问题


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

![](./Images/2.png)

图18-16 可视化结果

可以看到有很多直线的颜色被识别错了，比如最后一行的第1、3、5、6列，颜色错误。另外有一些大色块也没有识别对，比如第3行最后一列和第4行的头尾两个，都是大色块识别错误。也就是说，对两类形状上的颜色判断不准：

- 很细的线
- 很大的色块

这是什么原因呢？笔者分析：

1. 针对细直线，由于带颜色的像素点的数量非常少，被拆成向量后，这些像素点就会在1x784的矢量中彼此相距很远，特征不明显，很容易被判别成噪音；
2. 针对大色块，由于带颜色的像素点的数量非常多，即使被拆成向量，也会占据很大的部分，这样特征点与背景点的比例失衡，导致无法判断出到底哪个是特征点。

笔者认为以上两点是前馈神经网络在训练上的不稳定，以及最后准确度不高的主要原因。

当然有兴趣的读者也可以保留输入样本的三个彩色通道信息，把一个样本数据变成1x3x784=2352的向量进行试验，看看是不是可以提高准确率。

### 用卷积神经网络解决问题

下面我们看看卷积神经网络的表现。我们直接使用三通道的彩色图片，不需要再做数据转换了。

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

![](./Images/3.png)

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

可以看到我们在测试集上得到了96.3%的准确度，比前馈神经网络模型要高出很多，这也证明了卷积神经网络在图像识别上的能力。

图18-18是测试集中前64个测试样本的预测结果。


![](./Images/4.png)

图18-18 测试结果

在这一批的样本中，只有左下角的一个绿色直线被预测成蓝色了，其它的没发生错误。

## 八、实现几何图形分类

### 用卷积神经网络解决问题

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


![](./Images/shape_cnn_result.png)

图18-24 测试结果

绝大部分样本预测是正确的，只有最后一个样本，看上去应该是一个很扁的三角形，被预测成了菱形。

代码中需要data/ch18的.npz代码训练，这里并没有所以我在网上搜索的npz下载到本地进行训练。

## 九、实现几何图形及颜色分类

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

## 十、MNIST分类

模型搭建

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

## 十一、Fashion-MNIST分类

用卷积神经网络来解决问题

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

在测试集上得到91.12%的准确率，在测试集上的前几个样本的预测结果如图18-40所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistResult_cnn.png" ch="555" />

图18-40 测试结果

与前馈神经网络方案相比，这32个样本里只有一个错误，第4行最后一列，把第9类“短靴”预测成了“凉鞋”，因为这个样本中间有一个三角形的黑色块，与凉鞋的镂空设计很像。


## 总结知识

在本部分的学习中，我们逐步学习到了
1、卷积的前向计算
2、卷积的反向传播
3、池化的前向计算与反向传播
4、用代码实现一个卷积网络并训练一些实际数据。
5、介绍一些经典的卷积模型

   - AlexNet
   - ZF Net
   - VGG
   - GoogLeNet
   - ResNet
   - DenseNet

6、实现颜色分类
7、实现几何图形分类
8、实现几何图形及颜色分类
9、MNIST分类
10、Fashion-MNIST分类

## 心得体会

卷积神经网络是深度学习中的一个里程碑式的技术，有了这个技术，才会让计算机有能力理解图片和视频信息，才会有计算机视觉的众多应用。
我发现CNN现在在Computer Vision中有着举足轻重的地位。在课上老师讲解CNN实现几何图形及颜色分类的例子，我就对CNN有了很深的兴趣，我发现生活中CNN具体实例非常多，所以我急于弄清楚其原理，典型的卷积神经网络中，会至少包含以下几个层：卷积层、激活函数层、池化层、全连接分类层，每一层都会有其特有的功能。
卷积神经网络（CNN），是深度学习算法应用最成功的领域之一，卷积网络在本质上是一种输入到输出的映射，它能够学习大量的输入与输出之间的映射关系，而不需要任何输入和输出之间的精确的数学表达式，只要用已知的模式对卷积网络加以训练，网络就具有输入输出对之间的映射能力。通过老师讲解及网络查询，我对CNN有了更深更透彻的理解，目前计算机视觉是一个大热点，提前学习相关理论对我今后的发展及提升有很大的帮助。