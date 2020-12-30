# 远程遥控
![avatar](img\8.jpg)

将游戏手柄接收器插到电脑USB口

打开https://html5gamepad.com/网页，看下是否检查到遥控手柄。记下对应设备的INDEX数值为多少

按下遥控手柄按键对应的数值会变化。记下对应按键的名称。

![avatar](img\9.jpg)

运行代码记下手柄对应的按键

![avatar](img\10.jpg)

修改axes值对应不同的按钮

![avatar](img\11.jpg)

此代码会显示当前摄像头拍摄到的画面

![avatar](img\12.jpg)

通过按键控制小车拍照

## 代码分析
//收集数据
```
display(image)
display(widgets.HBox([free_count, free_button]))
display(widgets.HBox([blocked_count, blocked_button]))
```

//将数据集文件夹压缩为zip文件。
```
!zip -r -q dataset.zip dataset
```
## 总结
通过这部分实验代码可以实现手柄控制小车的移动。