# 目标跟踪

打开Notebook/object_following/，打开live_demo.ipynb文件

![avatar](img\20.jpg)

先将预先训练好的ssd_mobilenet_v2_coco.engine模型下载
![avatar](img\21.jpg)

运行程序后会输出检测到的coco对象。
![avatar](img\22.jpg)

运行程序后输出如图所示，被检测到的物体周围画着蓝色的方框，目标对象（jetbot跟随目标）将显示绿色边框
![avatar](img\23.jpg)

## 代码分析
//导入ObjectDetector类
```
from jetbot import ObjectDetector
model = ObjectDetector('ssd_mobilenet_v2_coco.engine')
```
//用相机输入
```
detections = model(camera.value)
print(detections)
```
//打印检测到的对象
```
from IPython.display import display
import ipywidgets.widgets as widgets
detections_widget = widgets.Textarea()
detections_widget.value = str(detections)
display(detections_widget)
```
//初始化小车
```
from jetbot import Robot
robot = Robot()
```

//断开摄像机并停止小车运行
```
import time
camera.unobserve_all()
time.sleep(1.0)
robot.stop()
```
//关闭相机，以便后续实验使用
```
camera.stop()
```
## 总结
这部分实验是通过代码让小车能够识别所检测的物体。