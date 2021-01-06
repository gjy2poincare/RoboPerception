# 设计报告

## 基于OpenVINO的图像处理、检测与识别完成车辆目标和行人目标

## 目录

1. 安装OpenVINO
2. 进行车辆检测
   2.1 基于demo_security_barrier_camera.bat
   2.2 分析
3. 进行车辆识别和行人检测
   3.1 找到如今最好的车辆识别模型之一----**YOLOV4模型**
   3.2 将此模型拷进openvino中重多的模型中来完成自己的课程设计
   3.3 将此模型通过实际的图片案例来测试该模型的实际性能
   **YOLOV4模型**
   3.4 分析
   3.5 VOLO-v4-tiny模型 
   3.6 分析
4. 在做本次设计报告的过程中遇到过的问题
5. 分析
6. 总结
7. 心得体会

## 1. 安装OpenVINO

Install the Intel® Distribution of OpenVINO™ toolkit core components

Install the dependencies:
1. 安装OpenVINO™工具包核心组件的Intel分发
2. 安装依赖项：

- 使用C++的MicrosoftVisualStudio*2019、2017或2015年与MSBuild

  ![节点](./image/6.14.png)

- CMake3.4或更高64位

  注: 如果要使用MicrosoftVisualStudio 2019，则需要安装CMake3.14。
  
  ![节点](./image/6.15.png)

- Python3.6.564位
 
  重要：作为此安装的一部分，请确保单击该选项将应用程序添加到PATH环境变量

3. 设置环境变量

   注：如果您将OpenVINO™的Intel分发版安装到非默认安装目录，请替换C:\Program Files (x86)\IntelSWTools使用安装软件的目录。

   在编译和运行OpenVINO™应用程序之前，必须更新几个环境变量。打开命令提示符，然后运行setupvars.bat批处理文件以临时设置环境变量：

  ```
  cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
  ```

  ```
  setupvars.bat
  ```

  ![节点](./image/6.16.png)

4. 配置模型优化器

- 选项1：同时为所有受支持的框架配置模型优化器：

  转到模型优化器先决条件目录。

  ```
  cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\install_prerequisites
  ```

  运行以下批处理文件来配置Caffe*、TensorFlow*、MXNet*、Kaldi*和ONNX*的模型优化器：

  ```
  install_prerequisites.bat
  ```

  

- 方案2：分别为每个框架配置模型优化器：

  转到模型优化器先决条件目录。

  ```
  cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\install_prerequisites
  ```

  For Caffe:

  ```
  install_prerequisites_caffe.bat
  ```

  For TensorFlow:

  ```
  install_prerequisites_tf.bat
  ```
  
  For MXNet:

  ```
  install_prerequisites_mxnet.bat
  ```
  
  For ONNX:

  ```
  install_prerequisites_onnx.bat
  ```
  
  For Kaldi:

  ```
  install_prerequisites_kaldi.bat
  ```

  注:运行时未安装Kaldi时，运行方案1就可运行，但系统未安装Kaldi


  ![节点](./image/6.17.png)


5. 跑二验证脚本以验证安装

  ```
  cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\demo\
  ```

  1. 运行图像分类验证脚本

  ```
  demo_squeezenet_download_convert_run.bat
  ```

    运行成功

  ![节点](./image/6.18.png)


## 2. 进行车辆检测

### 2.1 基于demo_security_barrier_camera.bat

  1. 设置环境变量
  
  需要把脚本里面的环境变量自己手动到系统里面更新一下,如果在import openvino出错了，很可能就是openvino的路径没有加到cvsdkpath去.

  ```
  cd C:\Program Files (x86)\IntelSWTools\openvino_2020.1.033\bin
  setupvars.bat
  ```
  ![节点](./image/1.png)

  2. 运行推理管道脚本

  ```
  demo_security_barrier_camera.bat
  ```
  
  运行后的车辆检测结果

  ![节点](./image/6.19.png)

  3. 更换图片进行推理管道脚本运行

  ![节点](./image/2.png)

  ![节点](./image/3.png)

### 2.2 分析

这个模型我们能很好的识别车辆和车牌，但是从左下角的黑色车和右上角的车辆没有识别可知，这个模型并不是最好的模型，而且这个程序只需要运行bat文件就可以得出结果，我们并不知道这个程序是如何得出的，所以后面我们将自己完成以下事情：

  1. 找到如今最好的车辆识别模型之一
  2. 将此模型拷进openvino中重多的模型中来完成自己的课程设计
  3. 将此模型通过实际的图片案例来测试该模型的实际性能


## 3. 进行车辆识别和行人检测

### 3.1 找到如今最好的车辆识别模型之一

<https://zhuanlan.zhihu.com/p/136253046>

通过此文可知，YOLO-v4算法是在原有YOLO目标检测架构的基础上，采用了近些年CNN领域中最优秀的优化策略，从数据处理、主干网络、网络训练、激活函数、损失函数等各个方面都有着不同程度的优化，虽没有理论上的创新，但是会受到许许多多的工程师的欢迎，各种优化算法的尝试。

所以本文的课程设计我将使用现在最新的YOLO-v4算法来进行

## 3.2 将此模型拷进openvino中重多的模型中来完成自己的课程设计

### 准备工作

- 环境

1. OpenVINO2020R4：<Https://docs.openvinotoolkit.org/latest/index.html>

2. Win或Ubuntu

3. TensorFlow 1.12.0

4. YOLOV 4：<Https://github.com/AlexeyAB/darknet>下载权重文件 或者通过百度网盘<https://pan.baidu.com/s/1abhBwbrX_flm_Mj07YlDnA> 提取码：965E

前面三者我们前期工作就已经装好，现在我们进行安装 4.YOLOV 4

我们先从<https://github.com/TNTWEN/OpenVINO-YOLOV4/blob/master/README.md>克隆到本地C盘

之所以拷到C盘是因为我们在后期win的cmd下，方便操作

然后我们将通过百度网盘下载权重文件

如图：

 ![节点](./image/4.png)

### 3.3 将此模型通过实际的图片案例来测试该模型的实际性能

### YOLOV4模型

1. 安装成功后我们打开cmd.exe

注：需要在管理员模式下，可以通过C盘下的C:/Windows/System32/cmd.exe反键进入管理员模型

![节点](./image/5.png)

2. 将YoLo-v4进openvino模型中

**将 weight --> pb**

```
#windows  default OpenVINO path

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4.weights --data_format NHWC
```

**convert_weights_pb.py**内部文件

``` py
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import yolo_v4
import yolo_v4_tiny
from PIL import Image, ImageDraw

from utils import load_weights, load_coco_names, detections_boxes, freeze_graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov4.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'output_graph', 'frozen_darknet_yolov4_model.pb', 'Frozen tensorflow protobuf model output path')

tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv4')
tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')



def main(argv=None):
    if FLAGS.tiny:
        model = yolo_v4_tiny.yolo_v4_tiny
    else:
        model = yolo_v4.yolo_v4

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], "inputs")

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, FLAGS.output_graph)

if __name__ == '__main__':
    tf.app.run()

```

**yolo_v4.py**内部文件，也就是模型

``` py
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(12, 16), (19, 36), (40, 28),
            (36, 75), (76, 55), (72, 146),
            (142, 110), (192, 243), (459, 401)]
@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]],
                               mode=mode)
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs



def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _yolo_res_Block(inputs,in_channels,res_num,data_format,double_ch=False):
    out_channels = in_channels
    if double_ch:
        out_channels = in_channels * 2
    net = _conv2d_fixed_padding(inputs,in_channels*2,kernel_size=3,strides=2)
    route = _conv2d_fixed_padding(net,out_channels,kernel_size=1)
    net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)

    for _ in range(res_num):
        tmp=net
        net = _conv2d_fixed_padding(net,in_channels,kernel_size=1)
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=3)
        #shortcut
        net = tmp+net

    net=_conv2d_fixed_padding(net,out_channels,kernel_size=1)

    #concat
    net=tf.concat([net,route],axis=1 if data_format == 'NCHW' else 3)
    net=_conv2d_fixed_padding(net,in_channels*2,kernel_size=1)
    return net

def _yolo_conv_block(net,in_channels,a,b):
    for _ in range(a):
        out_channels=in_channels/2
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)
        net = _conv2d_fixed_padding(net,in_channels,kernel_size=3)

    out_channels=in_channels
    for _ in range(b):
        out_channels=out_channels/2
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)

    return net


def _spp_block(inputs, data_format='NCHW'):
    return tf.concat([slim.max_pool2d(inputs, 13, 1, 'SAME'),
                      slim.max_pool2d(inputs, 9, 1, 'SAME'),
                      slim.max_pool2d(inputs, 5, 1, 'SAME'),
                      inputs],
                     axis=1 if data_format == 'NCHW' else 3)


def _upsample(inputs, out_shape, data_format='NCHW'):
    # tf.image.resize_nearest_neighbor accepts input in format NHWC
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    if data_format == 'NCHW':
        new_height = out_shape[2]
        new_width = out_shape[3]
    else:
        new_height = out_shape[1]
        new_width = out_shape[2]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    # back to NCHW if needed
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, name='upsampled')
    return inputs


def csp_darknet53(inputs,data_format,batch_norm_params):
    """
    Builds CSPDarknet-53 model.activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)
    """
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x:x* tf.math.tanh(tf.math.softplus(x))):
        net = _conv2d_fixed_padding(inputs,32,kernel_size=3)
        #downsample
        #res1
        net=_yolo_res_Block(net,32,1,data_format,double_ch=True)
        #res2
        net = _yolo_res_Block(net,64,2,data_format)
        #res8
        net = _yolo_res_Block(net,128,8,data_format)

        #features of 54 layer
        up_route_54=net
        #res8
        net = _yolo_res_Block(net,256,8,data_format)
        #featyres of 85 layer
        up_route_85=net
        #res4
        net=_yolo_res_Block(net,512,4,data_format)

    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
        ########
        net = _yolo_conv_block(net,1024,1,1)

        net=_spp_block(net,data_format=data_format)

        net=_conv2d_fixed_padding(net,512,kernel_size=1)
        net = _conv2d_fixed_padding(net, 1024, kernel_size=3)
        net = _conv2d_fixed_padding(net, 512, kernel_size=1)

        #features of 116 layer
        route_3=net

        net = _conv2d_fixed_padding(net,256,kernel_size=1)
        upsample_size = up_route_85.get_shape().as_list()
        net = _upsample(net, upsample_size, data_format)
        route= _conv2d_fixed_padding(up_route_85,256,kernel_size=1)

        net = tf.concat([route,net], axis=1 if data_format == 'NCHW' else 3)
        net = _yolo_conv_block(net,512,2,1)
        #features of 126 layer
        route_2=net

        net = _conv2d_fixed_padding(net,128,kernel_size=1)
        upsample_size = up_route_54.get_shape().as_list()
        net = _upsample(net, upsample_size, data_format)
        route= _conv2d_fixed_padding(up_route_54,128,kernel_size=1)
        net = tf.concat([route,net], axis=1 if data_format == 'NCHW' else 3)
        net = _yolo_conv_block(net,256,2,1)
        #features of 136 layer
        route_1 = net

    return route_1, route_2, route_3

def _get_size(shape, data_format):
    if len(shape) == 4:
        shape = shape[1:]
    return shape[1:3] if data_format == 'NCHW' else shape[0:2]


def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    num_anchors = len(anchors)
    predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                              stride=1, normalizer_fn=None,
                              activation_fn=None,
                              biases_initializer=tf.zeros_initializer())

    shape = predictions.get_shape().as_list()
    grid_size = _get_size(shape, data_format)
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes

    if data_format == 'NCHW':
        predictions = tf.reshape(
            predictions, [-1, num_anchors * bbox_attrs, dim])
        predictions = tf.transpose(predictions, [0, 2, 1])

    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(
        predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)
    return predictions




def yolo_v4(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v4 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :param with_spp: whether or not is using spp layer.
    :return:
    """

    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], data_format=data_format, reuse=reuse):

            #weights_regularizer=slim.l2_regularizer(weight_decay)
            #weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
        with tf.variable_scope('cspdarknet-53'):
            route_1, route_2, route_3 = csp_darknet53(inputs,data_format,batch_norm_params)

        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            biases_initializer=None,
                            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
            with tf.variable_scope('yolo-v4'):
                #features of y1
                net = _conv2d_fixed_padding(route_1,256,kernel_size=3)
                detect_1 = _detection_layer(
                    net, num_classes, _ANCHORS[0:3], img_size, data_format)
                detect_1 = tf.identity(detect_1, name='detect_1')

                #features of y2
                net = _conv2d_fixed_padding(route_1, 256, kernel_size=3,strides=2)
                net=tf.concat([net,route_2], axis=1 if data_format == 'NCHW' else 3)
                net=_yolo_conv_block(net,512,2,1)
                route_147 =net
                net = _conv2d_fixed_padding(net,512,kernel_size=3)
                detect_2 = _detection_layer(
                    net, num_classes, _ANCHORS[3:6], img_size, data_format)
                detect_2 = tf.identity(detect_2, name='detect_2')

                # features of  y3
                net=_conv2d_fixed_padding(route_147,512,strides=2,kernel_size=3)
                net = tf.concat([net, route_3], axis=1 if data_format == 'NCHW' else 3)
                net = _yolo_conv_block(net,1024,3,0)
                detect_3 = _detection_layer(
                    net, num_classes, _ANCHORS[6:9], img_size, data_format)
                detect_3 = tf.identity(detect_3, name='detect_3')

                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                detections = tf.identity(detections, name='detections')
                return detections
```

在cfg/coco.names下训练过的物体

```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
```

![节点](./image/7.png)

默认在当前文件夹下新建一个saved_model文件夹，里面是转换生成的文件：

![节点](./image/9.png)

如果是转换自己训练的数据集，则将coco.names和yolov4.weights替换成自己相应的文件就可以了。

**设置环境变量**

```
"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
```
![节点](./image/8.png)

**将 pb --> ir**

**切换到 OpenVINO-YOLOV4 目录下，将 pb 文件 转化为 xml 和 bin 文件。**

在python下运行mo.py，加入刚刚训练好的pb格式的frozen_darknet_yolov4_model.pb再转换配置yolov4.json，批为1，转换输入通道

```
python "C:\Program Files (x86)\IntelSWTools\openvino_2020.1.033\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels
```
![节点](./image/10.png)

Tip:
转换 IR 模型前一定要注意 op 算子的兼容性，以及对应平台的数据精度。 以下这个页面可以查询到具体信息，很多模型转化失败是由于还没有得到支持。

<https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html>

**运行模型**

```

python object_detection_demo_yolov4_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU

```

![节点](./image/11.png)

### 3.4 分析

相对于前面基于demo_security_barrier_camera.bat来运行的车辆检测，YOLO-v4模型的识别能力显然更加突出，不仅把车辆识别出来，而且还能进行行人检测，摩托，三轮车检测等等，而且在每一个被框起来的模型上我们都能看见是这个物体的概率。

### 3.5 VOLO-v4-tiny模型 

前面是YOLO-v4的模型 
用YOLO-v4-tiny同理，这里我们就不做详细步骤了挂上步骤，我们来看最终结果

```
#windows  default OpenVINO path

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny.weights --data_format NHWC --tiny

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolo_v4_tiny.json --batch 1 --reverse_input_channels

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU
```
注： 这里我就不附上源代码了，以免篇幅过长，如果需要可以在OpenVINO-YOLOV4中的vOlO_v4_tiny.py查看

运行结果如下：

![节点](./image/12.png)

### 3.6 分析

在运行过程中，我们能发现VOLO-v4-tiny识别的物体会更少，但是运行速度会比VOLO-v4快几倍，但相对于openvino给我们的案例demo_security_barrier_camera.bat来运行的车辆检测，强太多倍。


## 4. 在做本次设计报告的过程中遇到过的问题

### 问题一

在cmd终端下，无法mkdir文件

**解决**：需要在管理员模式下，可以通过C盘下的C:/Windows/System32/cmd.exe反键进入管理员模型

![节点](./image/5.png)

如果遇到问题时，首先考虑自己是否是在管理员模式下运行的cmd

### 问题二

在搜索一些问题时，可以关注一下openvino的中文社区，里面有很多不错的分享，尤其是那个openvino早餐栏目，我也是遇到很多坑后才发现很多问题真的很简单。

### 问题三

如果遇到在跑官网提示的一些验证demo的时候，出现一些错误，当时忘了截图，首先应该想到的是自己的cmd下的用户名到底有没有中文

![节点](./image/13.png)

### 问题四

一定要设置环境变量 setupvars.bat，需要把脚本里面的环境变量自己手动到系统里面更新一下,如果在import openvino出错了，很可能就是openvino的路径没有加到cvsdkpath去.

```
"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
```
![节点](./image/8.png)

## 5. 分析

- 基于demo_security_barrier_camera.bat进行车辆检测
  ![节点](./image/6.19.png)
  ![节点](./image/3.png)
这个模型我们能很好的识别车辆和车牌，但是从左下角的黑色车和右上角的车辆没有识别可知，这个模型并不是最好的模型，而且这个程序只需要运行bat文件就可以得出结果，我们并不知道这个程序是如何得出的，所以后面我自己完成了以下事情：

  1. 找到如今最好的车辆识别模型之一
  2. 将此模型拷进openvino中重多的模型中来完成自己的课程设计
  3. 将此模型通过实际的图片案例来测试该模型的实际性能

----------------------------------------------------------------------------

  用传统的步骤理解是：
  1. 根据自己的需求选择合适的网络并训练模型。
  2. 根据自己的训练模型需要配置 Mode Optimizer。
  3. 根据设置的模型参数运行 Model Optimizer, 生成相对应的 IR (主要是xml和bin)。

      注：xml 主要用来描述网络拓扑结构, bin 包括生成的 weights and biases 二进制数据
  4. 在实际应用场景种使用 Inference Engine 测试生成的 IR。
      **此步非常重要，自行生成的 IR 模型，一定要测试是否转换完全及是否成功**
  5. 在应用程序种调用 Inference Engine 相应接口，将生成的模型IR部署到实际环境中。
----------------------------------------------------------------------------
- 基于YOLOv4模型进行进行车辆识别和行人检测

我通过以上所说的3个步骤再具体话为传统的5个步骤进行此实验，基于YOLOV4模型运行出

![节点](./image/11.png)

如图可知

相对于前面基于demo_security_barrier_camera.bat来运行的车辆检测，YOLO-v4模型的识别能力显然更加突出，不仅把车辆识别出来，而且还能进行行人检测，摩托，三轮车检测等等，而且在每一个被框起来的模型上我们都能看见是这个物体的概率。

- 基于YOLOv4-tiny模型进行进行车辆识别和行人检测

![节点](./image/12.png)

在运行过程中，我们能发现VOLO-v4-tiny识别的物体会更少，但是运行速度会比VOLO-v4快几倍，但相对于openvino给我们的案例demo_security_barrier_camera.bat来运行的车辆检测，强太多倍。

## 6. 总结

- **1. OpenVINO 主要工作流程**

  OpenVINO 的主要工作流程如图：
  ![节点](./image/14.jpg)

  所以我们本次实验就是通过这些步骤得到了YOLO-v4和YOLO-v4-tiny的模型

  具体步骤如下

  1. 根据自己的需求选择合适的网络并训练模型。
  2. 根据自己的训练模型需要配置 Mode Optimizer。
  3. 根据设置的模型参数运行 Model Optimizer, 生成相对应的 IR (主要是xml和bin)。

      注：xml 主要用来描述网络拓扑结构, bin 包括生成的 weights and biases 二进制数据
  4. 在实际应用场景种使用 Inference Engine 测试生成的 IR。
      **此步非常重要，自行生成的 IR 模型，一定要测试是否转换完全及是否成功**
  5. 在应用程序种调用 Inference Engine 相应接口，将生成的模型IR部署到实际环境中。

- **2. Model optimizer （模型优化器）**
  Model Optimizer 是一个跨平台命令行工具，用于促进训练与具体实施平台中的过渡，主要是进行静态模型分析 以及根据配置参照自动调整深度模型。
  Model Optimizer 被用来设计成支持常用的框架（ Caffe, TensorFlow, MXNet, Kaldi, ONNX 等），相当于封装了一层，便于进行开发。

  Model Optimizer 主要工作流程：
  1、根据需要所用到的框架，配置 Model Optimizer。
  2、提供训练模型作为输入，包括网络拓扑以及参数。
  3、运行 Model Optimizer (根据选择的网络拓扑进行训练）。
  4、IR 作为 Model Optimizer 输出。

本文通过安装OpenVINO，来为后面的课程设计构建基础平台，再基于demo_security_barrier_camera.bat进行车辆检测,从而推出此案例并不能很好的检测车辆目标，以此找到如今最好的车辆识别模型之一----YOLOV4模型，然后将此模型拷进openvino中重多的模型中来完成自己的课程设计、基于YOLOV4模型进行OpenVINO主要工作流程步骤的设计来通过实际的图片案例来测试该模型的实际性能，最后再通过VOLO-v4-tiny的模型对YOLOv4模型进行简化，我们发现运行时间会缩短许多，但是车辆检测和行人检测的效果也会有所影响。

## 7. 心得体会

通过本文我掌握了对OpenVINO的安装，对OpenVINO工作原理的完全理解，也掌握了我们下载下来的openvino文件里面每个文件包具体是干什么的，比如Model_Zoo涵盖了大量的预训练模型用来行人识别、道路分割、表情识别等等，而且这些模型是通过优化的，可以直接拿来用于加速产品开发和部署。我也掌握了当我们cmake，make好这些预训练模型后它会build进入C:\Users\ASUS\Documents\Intel\OpenVINO中变成一个一个demo，我也通过自己的理解掌握了demo中的文件组成/内部构造，包括ir（FP16里包含xml和bin），cache和models模型比如我用到的YOLOv4和YOLOv4-tiny模型等等。然后我也掌握了如何通过找到的对自己例子最有利的模型在openvino中运行，来完成车辆检测，行人检测等等的应用。

当然这次课程设计我也遇到了许多问题比如：每次运行openvino程序时一定要记住配置环境变量，还有进入cmd时一定要是管理员模式，运行程序时最好不要有中文路径等等，这让我对每次通过openvino进行设计的时候不再再走这些以前的弯路。

而且通过这次课程设计，让我们把我们在人工智能概论和车载信息处理学到的知识进行结合运行，如果我没有学习这两门课程，直接来接触openvino等软件，对于openvino的工作原理，各个模型是用来做什么的肯定一脸懵，所以本次课程设计让我不仅对前两门课有了更深的认识，而且通过对这些知识的实际应用，让我很好的掌握了openvino这一软件的构造、运行原理，并在openvino上进行设计开发。

注：最后我会将OpenVINO-YOLOV4一整个模型文件附在另一个文件夹中