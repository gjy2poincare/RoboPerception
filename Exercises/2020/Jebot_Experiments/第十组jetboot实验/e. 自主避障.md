# 自主避障

打开Notebooks/collision_avoidance/。打开data_collection.ipynb文件。

![avatar](img\13.jpg)
运行此代码启用摄像头

![avatar](img\14.jpg)
![avatar](img\15.jpg)
拍摄足够数量的可通行与不可通行照片

打开Notebooks/collision_avoidance/，打开train_model.ipynb文件

![avatar](img\16.jpg)
![avatar](img\17.jpg)

运行程序训练神经网络，运行时间比较长。训练完成后，当前目录下会出现一个best_mode.pth文件

左侧打开Notebooks/collision_avoidance/。打开live_demo.ipynb文件

![avatar](img\18.jpg)
运行程序后会显示摄像头实时图像和一条滑条。互调表示遇到障碍物的概率，0.00表示前方没有障碍物，1.00表示前方哟障碍物需要转向避让。

![avatar](img\19.jpg)

适当降低小车速度避免速度太快直接撞上障碍物。

自主避障视频在同文件夹中，文件名为自主避障.mp4

## 总结
通过多次拍照来收集数据让小车进行神经网络训练，最后实现自主避障。