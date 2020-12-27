# 目标巡线
找到Notebooks/road_following/。打开data_collection.ipynb文件

![avatar](img\24.jpg)

运行程序后会显示一段youtube上的演示视频,但因为是国内用户无法观看

![avatar](img\25.jpg)

程序运行到此处会显示当前摄像头的测试图像。右边图像会显示一个绿点和蓝色线。用于表示小车应该运行的路线。
![avatar](img\26.jpg)

用手柄控制绿点和蓝色线
![avatar](img\27.jpg)

修改button值，设置对应的按键为拍摄图片按键，本次我们拍了14张图片
![avatar](img\28.jpg)

最后运行程序保存拍摄图片，当前目录下生成一个zip压缩文件。
![avatar](img\29.jpg)

找到Notebooks/road_following, 打开train_model.ipynb文件
![avatar](img\30.jpg)

下载神经模型
![avatar](img\31.jpg)

最后训练神经模型，当前目录下会生成best_steering_model_xy.pth文件
![avatar](img\35.jpg)

找到Notebooks/road_following，打开live_demo.ipynb文件
![avatar](img\32.jpg)

运行程序加载模型，打开摄像头实时显示图像，程序中有四个参数，可以通过拖动滑条改变参数的值。

输出展示jetbot的当前运行情况，x，y表示当前图像预测的x,y值，可以表示转动角度。speed表示jetbot直线速度，steering表示转向速度。
![avatar](img\33.jpg)

最后运行程序小车会转动，通过调节speed gain 滑条启动jetbot。（speed值尽量调小）

![avatar](img\34.jpg)

![avatar](img\36.jpg)

## 代码分析

//数据收集，显示实时图像提要
```
from jupyter_clickable_image_widget import ClickableImageWidget
DATASET_DIR = 'dataset_xy'
# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(DATASET_DIR)
except FileExistsError:
    print('Directories not created becasue they already exist')
camera = Camera()
# create image preview
camera_widget = ClickableImageWidget(width=camera.width, height=camera.height)
snapshot_widget = ipywidgets.Image(width=camera.width, height=camera.height)
traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=bgr8_to_jpeg)
# create widgets
count_widget = ipywidgets.IntText(description='count')
# manually update counts at initialization
count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))
def save_snapshot(_, content, msg):
    if content['event'] == 'click':
        data = content['eventData']
        x = data['offsetX']
        y = data['offsetY']
        
        # save to disk
        #dataset.save_entry(category_widget.value, camera.value, x, y)
        uuid = 'xy_%03d_%03d_%s' % (x, y, uuid1())
        image_path = os.path.join(DATASET_DIR, uuid + '.jpg')
        with open(image_path, 'wb') as f:
            f.write(camera_widget.value)
        
        # display saved snapshot
        snapshot = camera.value.copy()
        snapshot = cv2.circle(snapshot, (x, y), 8, (0, 255, 0), 3)
        snapshot_widget.value = bgr8_to_jpeg(snapshot)
        count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))
        
camera_widget.on_msg(save_snapshot)
data_collection_widget = ipywidgets.VBox([
    ipywidgets.HBox([camera_widget, snapshot_widget]),
    count_widget
])
display(data_collection_widget)
```
//关闭相机
```
camera.stop()
```
## 总结
这部分实验是通过代码先拍照让小车找到线，然后用多张照片让小车学习，最后小车成功巡线。