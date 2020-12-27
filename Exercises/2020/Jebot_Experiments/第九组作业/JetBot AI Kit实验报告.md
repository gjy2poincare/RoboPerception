
## 第九组
### 组长：吴海燕
### 组员：赵慧、王梦琪、王莹
# JetBot AI Kit实验报告
## 实验一 组装小车
### 实验目的
本实验需要按照说明书安装小车，并进行上电测试。

### 实验步骤
根据说明书组装小车
组装好了之后，就可以将开关拨到ON上电测试了

### 实验结果
![](https://ftp.bmp.ovh/imgs/2020/12/6d5eb21eb4ac25f0.jpg)

## 实验二 安装镜像
### 实验目的
安装镜像，将Jetbot连接到WiFi,并在web浏览器jupyterlab网页中连接小车

### 实验步骤
1. 烧写镜像
2. 启动Jetson Nano
   * 将SB卡插入Jetson Nano
   * 连接HDMI显示器，键盘和鼠标到Jetbot小车
   * 连接电源到Jetson，上电启动Jetson Nano
3. 连接JetBot到WiFi
   * 登录系统，Jetbot系统的默认用户名和密码均为jetbot
   * 点击系统右上角网络图标连接WiFi
   * 关机断电。两分钟后重新启动Jetson nano。启动的时候系统会自动连接WiFi，并同时在OLED显示器上显示IP地址     
4. Web浏览器连接Jetbot
   * Jetbot小车正常启动后在OLED屏幕上显示有小车的IP地址
   * 电脑连接上小车连接的WiFi，在浏览器输入Jetbot的IP地址。Port为8888
   * 首次打开需要输入用户名和密码登录。默认用户名和密码均为jetbot

### 实验结果
![](https://ftp.bmp.ovh/imgs/2020/12/c30e91ebc7bf8d04.jpg)

## 实验三、电机驱动
### 实验目的
运行basic_motion中的程序，使用网页按键来控制小车的前后左右移动，并通过"心跳"测验来保持小车的正常工作，测试电机是否运行正常。

### 实验步骤
* 在左侧打开/Notebooks/basic_motion/。打开basic_motion.ipynb文件
* 按照顺序及英文注释执行程序
* 运行语句robot.left(speed=0.3)时，小车会向左转圈
* 运行程序后，会输出左右两条滑条，拖动滑条可以改变左右轮的转速
![](https://ftp.bmp.ovh/imgs/2020/12/f2ac2ff46b645950.jpg)
* 运行到以下代码后，就可以通过网页按键来控制小车的前后左右移动了，右键该断代码，选择Create New View for Output将输出窗口作为新页面打开，按下其中的按钮来控制小车（在这期间，如果速度过快，可以通过调节前面的滑条来改变速度或者调节设置速度的数值）
![](https://ftp.bmp.ovh/imgs/2020/12/365558b18c49c020.jpg)
* 此段代码通过"心跳"来保存小车的正常工作，拖动滑条降低心跳频率后小车会停止跳动

### 实验结果
![](https://ftp.bmp.ovh/imgs/2020/12/88cdb66bf5c3aeba.gif)

## 实验四、远程遥控
### 实验目的
实现用遥控手柄控制小车的方向和拍照。

### 实验步骤
* 在左侧打开/Notebooks/teleoperation/。打开teleoperation.ipynb文件
* 将游戏手柄接收器插到电脑USB口
* 打开https://html5gamepad.com/网页，看下是否检查到遥控手柄。记下对应设备的INDEX数值为多少，我这里的数值是0。注意程序中的index需要修改为实际遥控手柄对应的，在刚才打开的网页中可以看到对应的INDEX。
* 运行到窗口会输出游戏手柄对应的按键。按下按键对应的按键图标会提示按键已按下。
![](https://ftp.bmp.ovh/imgs/2020/12/10764c93ab7926eb.jpg)
* 修改axes值或者buttons值来遥控小车
[![rsB9PJ.jpg](https://s3.ax1x.com/2020/12/22/rsB9PJ.jpg)](https://imgchr.com/i/rsB9PJ)

* 运行到Connect Camera...代码块时，窗口会显示当前摄像头拍摄到的画面 **插图**
* "心跳"检查模块，当小车断网时小车会自动停止
* 通过按键控制小车拍照，改变button可以选择不同的按键，我这里选择的是buttons[0]
* 运行程序之后即可通过遥控手柄来控制小车的方向和拍照了。 

### 实验结果
![](https://ftp.bmp.ovh/imgs/2020/12/a9e30b65a13f2275.gif)
![](https://ftp.bmp.ovh/imgs/2020/12/5132745242a1c356.gif)


## 实验五、自主避障
### 实验目的
实现小车的自主避障。  
要实现自主避障首先需要收集数据，通过摄像头拍摄各种图片，然后通过云端训练神经网络，最后通过训练的模型实现避障。
### 实验步骤
#### 第一步：在Jetbot上收集数据
* 左侧打开Notebooks/collision_avoidance/。打开data_collection.ipynb文件。
* 运行程序后出现如图所示界面，将小车放到不同的位置，如果前方没有障碍则点击add free. 如果小车前方有障碍就点击add blocked。尽可能多的收集各种类型的图片。拍摄到的图片会保存在dataset文件夹中。
![](https://ftp.bmp.ovh/imgs/2020/12/e388a41b5059235b.png)
* 运行程序打包图片成为dataset.zip压缩文件

#### 第二步：训练神经网络
* 在左侧打开Notebooks/collision_avoidance/，打开train_model.ipynb文件。
* 按顺序运行代码，第二个程序行最好不要运行，否则会覆盖刚刚压缩的dataset.zip文件
* 程序运行到Define the neural network会下载alexnet模型。下载程序后/home/hetbot/.torch/models目录下会出现alexnet-owt-4df8aa71.pth文件
* 最后运行程序训练神经网络，运行时间比较长。训练完成后，当前目录下会出现一个best_mode.pth文件。
![](https://ftp.bmp.ovh/imgs/2020/12/95a4acaf1b92e6a0.jpg)

#### 第三步：自主避障
* 在左侧打开Notebooks/collision_avoidance/。打开live_demo.ipynb文件。
* 运行程序后会显示摄像头实时图像和一条滑条。滑条表示遇到障碍物的概率，0.00表示前方没有障碍物，1.00表示前方有障碍物需要转向避让。
![](https://ftp.bmp.ovh/imgs/2020/12/6dace0db7cdf8b69.jpg)
* 在运行过程中需要调节小车的速度到合适的数值，足够小车转弯和前后运动，避免速度过快直接撞上障碍物或者速度过慢反应比较慢。

## 实验六、目标跟踪
### 实验目的
实现使用Jetbot小车来跟踪目标
### 实验步骤
* 在左侧打开Notebook/object_following/，打开live_demo.ipynb文件。
* 运行程序之前需要先将预先训练好的ssd_mobilenet_v2_coco.engine模型下载，解压后复制到当前文件夹目录下
![](https://ftp.bmp.ovh/imgs/2020/12/48c31f0372706bad.jpg)
* 运行第三段代码之前，需要将检测对象放到摄像头前面。运行程序后会输出检测到的coco对象。没有检测到对象则输出空数据[[ ]]。输出信息通过查表可知，如果同时检测到多个对象则输出多个信息。
![](https://ftp.bmp.ovh/imgs/2020/12/53b6eff3eb69efa7.jpg)
我这里检测到的是人
* 运行程序后，被检测到的物体周围画着蓝色的方框，目标对象（jetbot跟随目标）将显示绿色边框。
* 可以适当调小speed和turn gain的值，避免jetbot运行速度太快
* 当jetbot检测到目标会转向目标，如果被障碍物挡住jetbot会左转。

### 实验结果
![](https://ftp.bmp.ovh/imgs/2020/12/73171640d3c889b9.jpg)

## 实验七、目标巡线
### 实验目的
通过收集数据、巡线、自主运行来实现小车的自主巡线功能。
### 实验步骤
#### 在Jetbot上收集数据
* 在左侧中的Notebooks/road_following/。打开data_collection.ipynb文件。
* 按顺序执行程序，并观察程序运行结果
* 程序运行到此处会显示当前摄像头的测试图像。右边图像会显示一个绿点和蓝色线。用于表示小车应该运行的路线。为方便后面的操作，我们将输出窗口用新窗口打开。
![](https://ftp.bmp.ovh/imgs/2020/12/d2e0cae520f9b7af.jpg)
* 下面程序和游戏手柄遥控章节类似。
* 修改index为实际手柄对应的标号。修改axes为要控制的按键。我们这里的index为0
* 如果使用我们配置的游戏手柄，则需要按下HOME键，切换模式。使得指示灯为两个灯亮的状态。
* 修改button值，设置对应的按键为拍摄图片按键
[![rsr8US.jpg](https://s3.ax1x.com/2020/12/23/rsr8US.jpg)](https://imgchr.com/i/rsr8US)
* 下面开始收集数据，将小车放置到线的不同位置，控制手柄的方向键，将绿色点拖到黑线上。蓝色线即表示小车应该运行的方向。然后按下按键(我这里是buttons[5])拍照收集图片。尽可能多的收集各种情况的图片，count表示已经拍摄的图片数量。
* 最后运行程序保存拍摄图片，当前目录下生成一个zip压缩文件。
![](https://ftp.bmp.ovh/imgs/2020/12/0f951d9ee0bb9abe.jpg)

#### 训练神经网络
* 在左侧栏找到Notebooks/road_following, 打开train_model.ipynb文件。
* 下载神经模型
![](https://ftp.bmp.ovh/imgs/2020/12/8f816ad229390280.jpg)
* 最好训练神经模型，当前文件夹目录下会生成best_steering_model_xy.pth文件

#### 自主巡线
* 在左侧栏中找到Notebooks/road_following，打开live_demo.ipynb文件
* 运行程序加载模型，打开摄像头实时显示图像
* 程序中有四个参数，可以通过拖动滑条改变参数的值，使巡线效果更好
![](https://ftp.bmp.ovh/imgs/2020/12/4dbcddd0357fe93e.jpg)
* 此输出展示jetbot的当前运行情况，x，y表示当前图像预测的x,y值，可以表示转动角度。speed表示jetbot直线速度，steering表示转向速度。
* 最后运行程序小车会转动，通过调节speed gain 滑条启动jetbot。

## 实验八、ROS
### 实验目的
通过ROS系统操作Jetbot小车
### 实验步骤
#### 安装melodic版本ROS
```
# enable all Ubuntu packages
# 在旧版本的Ubuntu中使用以下命令行
sudo apt-add-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) universe"
sudo apt-add-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse"
sudo apt-add-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) restricted"

# add ROS repository to apt sources
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install ROS Base
sudo apt-get update
sudo apt-get install ros-melodic-ros-base
 
# add ROS paths to environment
sudo sh -c 'echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc'
```

安装python库
```
# pip should be installed
$ sudo apt-get install python-pip
# install Adafruit libraries
$ pip install Adafruit-MotorHAT
$ pip install Adafruit-SSD1306
```

添加i2c到user用户组
> sudo usermod -aG i2c $USER
#### 创建catkin工作空间
```
#创建catkin工作空间保存我们的ROS程序包

# create the catkin workspace
mkdir -p ~/workspace/catkin_ws/src
cd ~/workspace/catkin_ws
catkin_make
 
# add catkin_ws path to bashrc
sudo sh -c 'echo "source ~/workspace/catkin_ws/devel/setup.bash" >> ~/.bashrc'
关闭当前终端，重新打开一个新的终端。确认ROS是否安装成功

$ echo $ROS_PACKAGE_PATH 
/home/nvidia/workspace/catkin_ws/src:/opt/ros/melodic/share
```
#### 编译安装jetson-inference
```
# git and cmake should be installed
sudo apt-get install git cmake
 
# clone the repo and submodules
cd ~/workspace
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
 
# build from source
mkdir build
cd build
cmake ../
make
 
# install libraries
sudo make install
【注意】由于Jetson nano服务器均在国外，部分资源需要能够上外网才能可以获取，否则可能安装失败。另外需要安装一下库

sudo apt install libqt4-dev libglew-dev
```
对于下载失败(cmake ../出现错误)这个问题，可以使用以下命令将模型包下载到本地，再进行后续操作。
```
cd <jetson-inference>/data/networks/
tar -zxvf <model-archive-name>.tar.gz
```
#### 编译安装ros_deep_learning
```
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
#### 编译安装jetbot_os
```
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
#### 测试jetbit ROS
打开一个新终端，然后运行ros核心节点
> roscore
输出信息如下，则ros正常工作
![](https://www.waveshare.net/w/upload/thumb/6/60/JetBot_AI_Kit_Manual_35.jpg/900px-JetBot_AI_Kit_Manual_35.jpg)

运行电机节点
另外再打开一个终端，roscore节点不要关闭。运行如下命令启动jetbot_motors节点
> rosrun jetbot_ros jetbot_motors.py

再新建一个终端，输入rosnode list可以查看jetbot_motors节点是否启动。输入rostopic list可以查看jetbot_motor节点监听的话题

测试电机命令
打开一个新的终端，运行如下测试命令可以控制jetbot运动
```
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "forward"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "backward"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "left"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "right"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "stop"
```

使用OLED显示信息
新建一个终端运行OLED节点
> rosrun jetbot_ros jetbot_oled.py

jetbot_oled 监听/jetbot_oled/user_text话题，接受字符信息并显示。运行如下命令显示“HELLO”字符 rostopic pub /jetbot_oled/user_text std_msgs/String --once "HELLO!"

使用键盘控制jetbot移动
新建一个终端运行teleop_key节点
> rosrun jetbot_ros teleop_key.py
程序运行后通过键盘输入W、S、D、A四个按键控制jetbot前后左右移动

使用游戏摇杆控制jetbot移动
将游戏手柄的USB接收器插到jetbot 新建一个终端输入以下命令
> ls /dev/input
![](https://www.waveshare.net/w/upload/thumb/a/a0/JetBot_AI_Kit_Manual_37.jpg/600px-JetBot_AI_Kit_Manual_37.jpg)

## 实验总结
通过Jetbot小车的一系列实验，让我们对自动驾驶有了新的认识，对无人车也产生了浓厚的兴趣，更加的体验到了自己去实操和书本上的体会到的还是不一样的。在这个过程中尽管遇到好多问题，我们都非常积极主动的去完成，学到了很多Jetbot nano相关的知识，但是发现自己涉及到的领域也只不过是沧海一粟，在以后的学习中也要不断发现问题，解决问题，还是要注重团队合作，在团队中汲取灵感。