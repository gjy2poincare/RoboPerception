

学习总结：
今天经过一上午的学习和实验 ：配置opencv和调试vs，各种图像的处理以及py的安装，
pip的安装配置 学习到了很多 也提高了动手能力 对vs的综合应用 其中在配置opencv环境
时出现了很多波折 ，
1.在配置dubug|x64的链接器的输入目录时要添加
D:\Path\opencv\build\x64\vc15\lib\opencv_world411d.lib不能opencv_world411d.lib
和opencv_world411.lib同时在一起 否则会报错
其他的按照步骤一一配置即可 运行代码 显示图片就成了。
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

void main()
{
	Mat srcImage = imread("C:\\Users\\Administrator\\Desktop\\q.jpg");
	imshow("imm", srcImage);
	waitKey(0);
}
图片位置一定要写对。
2.学会配置py和pip和调试opencv2 运用power shell
3.通过编写代码对图片进行各种处理。以下是腐蚀代码：
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
int main()
{
	Mat srcImage = imread("C:\\Users\\lenovo\\Desktop\\Project\\1.jpg");
	imshow("imm", srcImage);
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	Mat dstImage;
	erode(srcImage, dstImage, element);
	imshow("imm", dstImage);
	waitKey(0);
	return 0;
}



![](q.jpg)