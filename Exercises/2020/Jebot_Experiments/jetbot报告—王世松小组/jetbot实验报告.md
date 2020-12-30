

>>> #  jetbot实验  第六组   组员：王世松  徐子晶  孔名遥  张容玮

## 一.  Jetbot AI kit 安装

### 1.按照组装手册上的安装步骤安装小车

![](./jetbot/1.jpg)
<center>组件清单</center>

### 2.安装过程

![](jetbot/7.jpg)
<center>安装</center>

![](jetbot/8.jpg)
<center>安装</center>

![](jetbot/9.jpg)
<center>安装</center>

![](jetbot/2.jpg)
<center>安装</center>

![](jetbot/3.jpg)
<center>安装</center>


![](jetbot/4.jpg)
<center>安装</center>

![](jetbot/5.jpg)
<center>安装</center>


>>> ## 二、安装成功

![](jetbot/6.jpg)
<center>安装_6</center>

### 3.为小车安装操作系统
进入快捷入口的Terminal打开一个终端窗口，输入以下指令：
- git clone https://github.com/waveehare/jetbot
- cd jetbot
- sudo python3 setup.py install

![](jetbot/11.png)
<center>安装操作系统</center>

将小车连接到与电脑相同的局域网下：
![](jetbot/12.png)
<center>连接WiFi</center>

![](jetbot/13.png)
<center>烧写成功</center>

### 4.在电脑上进入jupyter
连接到同一局域网后，在电脑的浏览器中输入IP地址加相应端口号，即可进入jupyter
密码：jetbot

![](./jetbot/22.png)
<center>进入jupyter</center>

>>> ## 三、实验正式操作
### 实验二.连接手柄
- 打开Notebooks/teleoperation路径下的teleoperation.ipynb文件
- 观察手柄前面显示面板，如果指示灯没有亮，按下HOME键
- 通过HOME键切换两种工作模式：
- 1.只有一个指示灯亮时，摇杆输出为0；
- 2.两个指示灯亮时，摇杆输出模拟值（为了后面数据收集使用）


![](jetbot/14.jpg)
<center>连接手柄</center>

![](jetbot/80.png)
<center>查看手柄数据</center>


### 实验三.连接相机并测试

- 创建相机实例
![](jetbot/16.png)
<center>相机测试</center>

### 实验四.电机驱动

- 远程终端控制电机
![](./jetbot/17.png)
<center>控制电机</center>

![](./jetbot/18.jpg)
<center>摄像头功能</center>

- 设置滑块控制电机
![](./jetbot/18.png)
<center>控制电机</center> 

![](./jetbot/19.jpg)
<center>电机运转</center>

- 设置按钮控制电机

![](./jetbot/20.jpg)
<center>控制电机</center>

- Heartbeat Killswitch检测
  
![](./jetbot/21.jpg)
<center>控制电机</center>

### 实验五.自主避障

`通过前面的测试之后，就可以开始自主避障的实验了`

- 打开摄像头
![](./jetbot/23.png)
<center>打开摄像头</center>

- 运行程序进行数据采集
![](./jetbot/24.png)
<center>数据采集</center>

- 下载神经网络
![](./jetbot/27.jpg)
<center>下载神经网络</center>

- 训练神经网络
![](./jetbot/28.jpg)
<center>训练神经网络</center>

- 运行自主避障程序

![](./jetbot/26.png)
<center>打开摄像头并启动电机</center>

![](./jetbot/25.png)
<center>自主避障中</center>


### 实验六.目标跟踪
这一节使用jetbot跟踪目标，我们使用预先训练好的coco数据集神经网络，可以检测90种不同的物体

①  在浏览器地址栏输入http://<jetbot_ip_address>:8888连接到小车，左侧打开Notebook/object_following/，打开live_demo.ipynb文件

② 运行程序之前需要先将预先训练好的ssd_mobilenet_v2_coco.engine模型下载，解压后复制到当前文件夹目录下

③ 需要注意的时候，本章程序需要用到上一章自主避障中建立的模块，小车需要再同一个环境中进行。

④ 运行此段代码之前，需要将检测对象放到摄像头前面。运行程序后会输出检测到的coco对象。没有检测到对象则输出空数据[[ ]]。输出信息通过查表可知检测到了苹果，苹果的ID为53。如果同时检测到多个对象则输出多个信息。

#### 运行以下代码：
```python
from IPython.display import display
import ipywidgets.widgets as widgets

detections_widget=widgets.Textarea()

detections_widget.value=str(detections)

display(detections_widget)
```
![](jetbot/30.png)
<center>目标跟踪</center>

由于周围环境中有其他物体干扰，识别物体的标签号为76，图片中看出有多个蓝色方框，说明识别比较混乱，旁边的绿色方框表明小车即将追踪的物体所在方向

可以适当调小speed、turn gain的值，避免jebot运行速度过快

## 实验七.目标巡线

### （一）在JetBot上收集数据
① 在浏览器地址栏输入http://<jetbot_ip_address>:8888连接到小车，找到Notebooks/road_following/。打开data_collection.ipynb文件

②根据教程介绍，运行程序后会显示一段youtube上的演示视频，但由于无法连接外网，不能播放

③会显示当前摄像头的测试图像。右边图像会显示一个绿点和蓝色线。用于表示小车应该运行的路线。为方便后面的操作，我们将输出窗口用新窗口打开。
![](jetbot/123.png)
<center>目标巡线_1</center>

④下面的程序和游戏手柄遥控章节类似。修改index为实际手柄对应的标号。修改axes为要控制的按键。
![](jetbot/31.png)
<center>目标巡线_2</center>

<font color=orange>**【注意】**</font>
此处的axes按键必须为模拟按键，即可以输出小数。如果使用我们配置的游戏手柄，则需要按下HOME键，切换模式。使得指示灯为两个灯亮的状态

⑤收集数据
将小车放置到线的不同位置，控制手柄的方向键，将绿色点拖到黑线上。蓝色线即表示小车应该运行的方向。然后按下按键拍照收集图片。尽可能多的收集各种情况的图片，count表示已经拍摄的图片数量
![](./jetbot/31.png)
<center>收集照片_1</center>

<font color=orange>**【注意】**</font>使用手柄控制绿点位置可能操作很不方便，可以通过拖动steering和throttle两个滑条来改变位置。

- 运行如下程序保存拍摄图片，当前目录下生成一个zip压缩文件
```PYTHON
def timestr():
return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

!zip -r -q road_following_{DATASET_DIR}_{timestr()}.zip {DATASET_DIR}
```
### （二）训练神经网络

①在浏览器地址栏输入http://<jetbot_ip_address>:8888连接到小车，找到Notebooks/road_following, 打开train_model.ipynb文件。

<font color =green>**ps:**</font>
如果使用的是上一节手机到的数据，不需要解压，图片文件已经在当前目录下。
如果需要另外解压文件，需要将road_following.zip改成对应的ZIP文件名，否则会提示文件不存在

② 下载神经模型
![](jetbot/32.png)
<center>下载模型</center>

③训练神经模型
![](jetbot/33.png)
<center>训练模型</center>

### （三）自主巡线

①在浏览器地址栏输入http://<jetbot_ip_address>:8888连接到小车，找到Notebooks/road_following，打开live_demo.ipynb文件

②运行程序加载模型，打开摄像头实时显示图像

③程序中有四个参数，可以通过拖动滑条改变参数的值，如果需要实现巡线功能需要根据实际情况调试参数，使巡线的效果更好。

![](jetbot/34.png)
<center>巡线参数_1</center>

![](jetbot/35.png)
<center>巡线参数_2</center>

④ 此输出展示jetbot的当前运行情况，x，y表示当前图像预测的x,y值，可以表示转动角度。speed表示jetbot直线速度，steering表示转向速度。

⑤ 最后运行程序小车会转动，通过调节速度增益(speed gain)滑条启动jetbot. 

<font color=orange>**【注意】**</font>速度(speed)值尽量小一点，否则速度太快容易冲出线外。当jetbot沿线运行，但是左右摆动幅度太大，可以减小转向速度(steering)的值，是jetbot运动更加平滑，摆动幅度更小。

## 11. ROS
<font color=green>**前提条件：**</font>已下载jetbot镜像

⑴ 安装melodic版本ROS
Jetson Nano提供的系统镜像是基于18.04版本Ubuntu系统，支持直接用apt安装ROS.使用下面的指令，按顺序安装，
```python
sudo apt-add-repository universe
sudo apt-add-repository multiverse
sudo apt-add-repository restricted
 
# add ROS repository to apt sources
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
 
# install ROS Base
sudo apt-get update
sudo apt-get install ros-melodic-ros-base
 
# add ROS paths to environment
sudo sh -c 'echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc'
```
![](jetbot/36.png)
<center>运行结果_1</center>

![](jetbot/37.png)
<center>运行结果_2</center>

安装python库
```python
# pip should be installed
$ sudo apt-get install python-pip
# install Adafruit libraries
$ pip install Adafruit-MotorHAT
$ pip install Adafruit-SSD1306
```
添加i2C到用户组
```python
sudo usermod -aG i2c $USER
```
![](jetbot/38.png)
<center>运行结果_3</center>

⑵创建catkin工作空间
保存我们的ROS程序包
```python
# create the catkin workspace
mkdir -p ~/workspace/catkin_ws/src
cd ~/workspace/catkin_ws
catkin_make
 
# add catkin_ws path to bashrc
sudo sh -c 'echo "source ~/workspace/catkin_ws/devel/setup.bash" >> ~/.bashrc'
```
关闭当前终端，重新打开一个新的终端。确认ROS是否安装成功
```python
$ echo $ROS_PACKAGE_PATH 
/home/nvidia/workspace/catkin_ws/src:/opt/ros/melodic/share
```
⑶编译安装jetson-inference
```python
# git and cmake should be installed
sudo apt-get install git cmake
 
# clone the repo and submodules
cd ~/workspace
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
```
![](jetbot/39.png)
<center>运行结果_4</center>

![](jetbot/40.png)
<center>运行结果_5</center>

```python
# build from source
mkdir build
cd build
cmake ../
make
 
# install libraries
sudo make install
```

![](jetbot/41.png)
<center>运行结果_6</center>
选择‘OK’后，安装python3.6，小车内部运行相应程序，等待几秒即为下图所示：

![](jetbot/42.png)
<center>运行结果_7</center>

![](jetbot/43.png)
<center>运行结果_8</center>

<font color=orange>**【注意】**</font>由于Jetson nano服务器均在国外，部分资源需要能够上外网才能可以获取，否则可能安装失败。另外需要安装一下库
```python
sudo apt install libqt4-dev libglew-dev
```
⑷ 编译安装ros_deep_learning
```python
# install dependencies
sudo apt-get install ros-melodic-vision-msgs ros-melodic-image-transport ros-melodic-image-publisher
 
# clone the repo
cd ~/workspace/catkin_ws/src
git clone https://github.com/dusty-nv/ros_deep_learning
 
# make ros_deep_learning
cd ../    # cd ~/workspace/catkin_ws
catkin_make
 
# confirm that the package can be found
rospack find ros_deep_learning
/home/nvidia/workspace/catkin_ws/src/ros_deep_learning
```
![](jetbot/44.png)
<center>运行结果_9</center>

⑸编译安装jetbot_ros
```python
# clone the repo
cd ~/workspace/catkin_ws/src
git clone https://github.com/waveshare/jetbot_ros
 
# build the package
cd ../    # cd ~/workspace/catkin_ws
catkin_make
 
# confirm that jetbot_ros package can be found
rospack find jetbot_ros
/home/nvidia/workspace/catkin_ws/src/jetbot_ros
```
![](jetbot/45.png)
<center>运行结果_10</center>

![](jetbot/46.png)
<center>运行结果_11</center>

⑹测试jetbot ROS
打开一个新终端，然后运行ros核心节点
输入以下命令：
```
roscore
```


<font color=orange>**【注意】**</font>roscore核心节点必须保持运行状态，否则其他所有的节点都不能工作

⑺测试电机命令
打开一个新的终端，运行如下测试命令可以控制jetbot运动
```python
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "forward"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "backward"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "left"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "right"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "stop"
```

![](jetbot/47.png)
<center>jetbot命名 </center>

⑻使用OLED显示信息
新建一个终端运行OLED节点
```
rosrun jetbot_ros jetbot_oled.py
```
jetbot_oled 监听/jetbot_oled/user_text话题，接受字符信息并显示。运行如下命令显示“HELLO”字符 rostopic pub /jetbot_oled/user_text std_msgs/String --once "HELLO!"

⑼使用键盘控制jetbot移动
新建一个终端运行teleop_key节点
```
rosrun jetbot_ros teleop_key.py
```
程序运行后通过键盘输入W、S、D、A四个按键控制jetbot前后左右移动

⑽使用游戏摇杆控制jetbot移动
将游戏手柄的USB接收器插到jetbot 新建一个终端输入以下命令
```
ls /dev/input
```
其中的js0就是代表游戏摇杆手柄，输入如下命令测试设备是否正常工作
```
sudo jstest /dev/input/js0
```
![](jetbot/48.png)
<center>运行结果_14</center>
遥控手柄按下不同的按键，对应的按键值会改变.

⑾安装joy包，并启动joy节点。要获得通过ROS发布的摇杆数据，我们需要启动joy节点。
```
sudo apt-get install ros-melodic-joy
rosrun joy joy_node
```
![](jetbot/49.png)
<center>运行结果_15</center>

⑿新建一个终端运行如下命令显示joy节点信息
```
rostopic echo joy
```
当按下游戏摇杆按下时，会输出类似如下的信息.
![](jetbot/joy.png)
<center>joy</center>

其中axes[ ]表示摇杆的数据，buttons[ ]表示按键的数据。

此时操作游戏手柄虽然可以显示ros发布的数据，但是要不能遥控jetbot。还需要运行teleop_joy节点，接收joy节点发布的话题信息，转换为/jetbot_motor/ cmd_str话题再发给jetbot_motors节点控制电机运动。新建一个终端运行如下命令启动终端teleop_joy节点。
```
rosrun jetbot_ros teleop_joy.py
```

安装游戏手柄的A键，然后控制上下左右方向键既可以控制jetbot移动，松开A键，jetbot则停止。

修改jetbot_ros/scripts/teleop_joy.py程序中 axes[ ]和 buttons[ ] 的值可以设置不同的按键控制。

⒀Jetbot 在~/.bashrc 文件最后面添加下面两行语句
```
export ROS_HOSTNAME=jetbot.local
export ROS_MASTER_URI=http://jetbot.local:11311
```
Ubuntu 系统在 ~/.bashrc 文件最后面添加下面两行语句
```
export ROS_HOSTNAME=ubuntu.local
export ROS_MASTER_URI=http://jetbot.local:11311
```
这样jetbot 和ubuntu虚拟机就可以通过ros节点通讯了，这样就可以在电脑端通过ubuntu虚拟机运行rqt_imge_view命令查看jetbot摄像头的实时图像了。
![](jetbot/60.png)
<center>摄像头输出的数据流</center>

### 实验结果输出

![](jetbot/70.jpg)
<center>最终结果</center>


>>> ## 四、实验总结与思考

### 9. 实验小结

- ### 实验步骤简述

本次实验是对Jetbot小车操作的实验，组装小车，将小车的零件按到说明书所指示的内容，依步骤安装起来。然后再通过网上下载系统软件，拷贝至小车中，最后通过群里分享的教程，以及小车系统中自带的notebook为基础来展开实验。

- ### 实验中遇到的问题及解决方式

1.在打开摄像头之后，因为小车的算力不足，导致摄像头卡住。
2.在执行浏览器上的代码时，也会进行报错，重启能解决大部分问题。
3.在下载神经网络模型步骤，模型不能下载完全，重装系统可以解决此类问题。


- ### Jetbot实验收获：
1.xshell的使用。
2.notebook的使用。
3.Ubuntu系统的使用。
##### 4.增进了小组同学的团队性，通过本次实验，增进了同学们的友谊，我作为组长，认感到欣慰和高兴。
