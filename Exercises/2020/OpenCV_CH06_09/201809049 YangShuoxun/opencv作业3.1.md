# OpenCV编程入门

## 六、图像处理

### 6.1 线性滤波

#### 平滑处理
平滑处理(smoothing）也称模糊处理(bluring)，是一种简单且使用频率很高的图像处理方法。平滑处理的用途有很多，最常见的是用来减少图像上的噪点或者失真。在涉及到降低图像分辨率时，平滑处理是非常好用的方法。

#### 图像滤波与滤波器
图像滤波,指在尽量保留图像细节特征的条件下对目标图像的噪声进行抑制,是图像预处理中不可缺少的操作，其处理效果的好坏将直接影响到后续图像处理和分析的有效性和可靠性。

消除图像中的噪声成分叫作图像的平滑化或滤波操作。信号或图像的能量大部分集中在幅度谱的低频和中频段，而在较高频段，有用的信息经常被噪声淹没。因此一个能降低高频成分幅度的滤波器就能够减弱噪声的影响。

图像滤波的目的有两个:
- 一个是抽出对象的特征作为图像识别的特征模式;
- 另一个是为适应图像处理的要求，消除图像数字化时所混入的噪声。

滤波处理的要求也有两条:
- 一是不能损坏图像的轮廓及边缘等重要信息;
- 二是使图像清晰视觉效果好。

滤波器，一种形象的比喻是:可以把滤波器想象成一个包含加权系数的窗口，当使用这个滤波器平滑处理图像时，就把这个窗口放到图像之上，透过这个窗口来看我们得到的图像。

滤波器的种类有很多，在新版本的OpenCV中，提供了如下5种常用的图像平滑处理操作方法，它们分别被封装在单独的函数中，使用起来非常方便。
- 方框滤波———BoxBlur函数
- 均值滤波（邻域平均滤波）——Blur函数
- 高斯滤波———GaussianBlur函数
- 中值滤波——medianBlur函数
- 双边滤波———bilateralFilter函数

#### 线性滤波器
线性滤波器经常用于剔除输入信号中不想要的频率或者从许多频率中选择一个想要的频率。

几种常见的线性滤波器：
- 低通滤波器:允许低频率通过;
- 高通滤波器:允许高频率通过;
- 带通滤波器:允许-定范围频率通过;
- 带阻滤波器:阻止- -定范围频率通过并且允许其他频率通过;
- 全通滤波器:允许所有频率通过，仅仅改变相位关系;
- 陷波滤波器( Band-Stop Filter): 阻止一个狭窄频率范围通过，是一 种特殊带阻滤波器。

#### 滤波和模糊
滤波可分低通滤波和高通滤波两种:高斯滤波是指用高斯函数作为滤波函数的滤波操作，至于是不是模糊，要看是高斯低通还是高斯高通，低通就是模糊，高通就是锐化。

#### 领域算子与线性领域滤波
邻域算子（局部算子）是利用给定像素周围的像素值的决定此像素的最终输出值的一种算子。而线性邻域滤波就是一种常用的邻域算子，像素的输出值取决于输入像素的加权和。

线性邻域滤波算子，即用不同的权重去结合一个小邻域内的像素，来得到应有的处理效果。

#### 方框滤波( box Filter )
方框滤波（box Filter）被封装在一个名为boxblur 的函数中，即 boxblur函数的作用是使用方框滤波器(box filter）来模糊一张图片，从 src输入，从 dst输出。函数原型如下。
```C++
 void boxFilter(InputArray src,outputArray dst, int ddepth,sizeksize,Point anchor=Point (-1,-1), boolnormalize=true,int borderType=BORDER_DEFAULT )
```

#### 均值滤波
均值滤波，是最简单的一种滤波操作，输出图像的每一个像素是核窗口内输入图像对应像素的平均值(所有像素加权系数相等)，其实说白了它就是归一化后的方框滤波。我们在下文进行源码剖析时会发现，blur函数内部中其实就是调用了一下boxFilter。blur函数的原型如下。
```C++
void blur (InputArray src，outputArraydst,size ksize,Pointanchor=Point (-1,-1), int borderType=BORDER_DEFAULT )
```

#### 高斯滤波
GaussianBlur函数的作用是用高斯滤波器来模糊一张图片，对输入的图像src进行高斯滤波后用dst输出。它将源图像和指定的高斯核函数做卷积运算，并且支持就地过滤(In-placefiltering)。
```C++
void GaussianBlur(InputArray src,outputArray dst, Size ksize,double sigmax,double sigmaY=0, intborderType=BORDER_DEFAULT )
```

### 程序：图像线性滤波综合
代码：
```C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage1, g_dstImage2, g_dstImage3;//存储图片的Mat类型
int g_nBoxFilterValue = 3;  //方框滤波参数值
int g_nMeanBlurValue = 3;  //均值滤波参数值
int g_nGaussianBlurValue = 3;  //高斯滤波参数值

//四个轨迹条的回调函数
static void on_BoxFilter(int, void*);		//均值滤波
static void on_MeanBlur(int, void*);		//均值滤波
static void on_GaussianBlur(int, void*);			//高斯滤波

int main()
{
	//改变console字体颜色
	system("color 5F");

	// 载入原图
	g_srcImage = imread("1.jpg", 1);
	if (!g_srcImage.data) { printf("Oh，no，读取srcImage错误~！ \n"); return false; }

	//克隆原图到三个Mat类型中
	g_dstImage1 = g_srcImage.clone();
	g_dstImage2 = g_srcImage.clone();
	g_dstImage3 = g_srcImage.clone();

	//显示原图
	namedWindow("【<0>原图窗口】", 1);
	imshow("【<0>原图窗口】", g_srcImage);


	//=================【<1>方框滤波】==================
	//创建窗口
	namedWindow("【<1>方框滤波】", 1);
	//创建轨迹条
	createTrackbar("内核值：", "【<1>方框滤波】", &g_nBoxFilterValue, 40, on_BoxFilter);
	on_BoxFilter(g_nBoxFilterValue, 0);
	//================================================

	//=================【<2>均值滤波】==================
	//创建窗口
	namedWindow("【<2>均值滤波】", 1);
	//创建轨迹条
	createTrackbar("内核值：", "【<2>均值滤波】", &g_nMeanBlurValue, 40, on_MeanBlur);
	on_MeanBlur(g_nMeanBlurValue, 0);
	//================================================

	//=================【<3>高斯滤波】=====================
	//创建窗口
	namedWindow("【<3>高斯滤波】", 1);
	//创建轨迹条
	createTrackbar("内核值：", "【<3>高斯滤波】", &g_nGaussianBlurValue, 40, on_GaussianBlur);
	on_GaussianBlur(g_nGaussianBlurValue, 0);
	//================================================


	//输出一些帮助信息
	cout << endl << "\t运行成功，请调整滚动条观察图像效果~\n\n"
		<< "\t按下“q”键时，程序退出。\n";

	//按下“q”键时，程序退出
	while (char(waitKey(1)) != 'q') {}

	return 0;
}

static void on_BoxFilter(int, void*)
{
	//方框滤波操作
	boxFilter(g_srcImage, g_dstImage1, -1, Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1));
	//显示窗口
	imshow("【<1>方框滤波】", g_dstImage1);
}

static void on_MeanBlur(int, void*)
{
	//均值滤波操作
	blur(g_srcImage, g_dstImage2, Size(g_nMeanBlurValue + 1, g_nMeanBlurValue + 1), Point(-1, -1));
	//显示窗口
	imshow("【<2>均值滤波】", g_dstImage2);
}

static void on_GaussianBlur(int, void*)
{
	//高斯滤波操作
	GaussianBlur(g_srcImage, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
	//显示窗口
	imshow("【<3>高斯滤波】", g_dstImage3);
}

```

实验结果：
![avatar](\picture\23.线性滤波.png)

### 6.2 非线性滤波

#### 中值滤波: medianBlur函数
中值滤波（Median filter）是一种典型的非线性滤波技术，基本思想是用像素点邻域灰度值的中值来代替该像素点的灰度值，该方法在去除脉冲噪声、椒盐噪声的同时又能保留图像的边缘细节。

中值滤波是基于排序统计理论的一种能有效抑制噪声的非线性信号处理技术，其基本原理是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替，让周围的像素值接近真实值，从而消除孤立的噪声点。这对于斑点噪声 ( speckle noise）和椒盐噪声( salt-and-pepper noise）来说尤其有用，因为它不依赖于邻域内那些与典型值差别很大的值。中值滤波器在处理连续图像窗函数时与线性滤波器的工作方式类似，但滤波过程却不再是加权运算。

中值滤波在一定的条件下可以克服常见线性滤波器，如最小均方滤波、方框滤波器、均值滤波等带来的图像细节模糊，而且对滤除脉冲干扰及图像扫描噪声非常有效，也常用于保护边缘信息。保存边缘的特性使它在不希望出现边缘模糊的场合也很有用，是非常经典的平滑噪声处理方法。

medianBlur函数使用中值滤波器来平滑（模糊）处理一张图片，从src输入，结果从 dst输出。对于多通道图片，它对每一个通道都单独进行处理，并且支持就地操作(In-placeoperation)。函数原型如下。
```C++
void medianBlur (InputArray src,outputArray dst,int ksize)
```

#### 双边滤波: bilateralFilter函数
双边滤波(Bilateral filter）是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折中处理，同时考虑空域信息和灰度相似性，达到保边去噪的目的，具有简单、非迭代、局部的特点。

双边滤波器的好处是可以做边缘保存（edge preserving)。以往常用维纳滤波或者高斯滤波去降噪，但二者都会较明显地模糊边缘，对于高频细节的保护效果并不明显。双边滤波器顾名思义，比高斯滤波多了一个高斯方差sigma-d，它是基于空间分布的高斯滤波函数，所以在边缘附近，离得较远的像素不会对边缘上的像素值影响太多，这样就保证了边缘附近像素值的保存。但是，由于保存了过多的高频信息，对于彩色图像里的高频噪声，双边滤波器不能够干净地滤掉，只能对于低频信息进行较好地滤波。

bilateralFilter函数的作用是用双边滤波器来模糊处理一张图片，由 src输入图片，结果于dst输出。函数原型如下。
```C++
void bilateralFilter(InputArray src,OutputArraydst,int d,double sigmacolor，double sigmaSpace, int borderType=BORDER_DEFAULT)
```

### 程序：五种图像滤波综合
代码：
```C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

//-----------------------------------【命名空间声明部分】---------------------------------------
//		描述：包含程序所使用的命名空间
//-----------------------------------------------------------------------------------------------  
using namespace std;
using namespace cv;


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage1, g_dstImage2, g_dstImage3, g_dstImage4, g_dstImage5;
int g_nBoxFilterValue = 6;  //方框滤波内核值
int g_nMeanBlurValue = 10;  //均值滤波内核值
int g_nGaussianBlurValue = 6;  //高斯滤波内核值
int g_nMedianBlurValue = 10;  //中值滤波参数值
int g_nBilateralFilterValue = 10;  //双边滤波参数值


//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------
//轨迹条回调函数
static void on_BoxFilter(int, void*);		//方框滤波
static void on_MeanBlur(int, void*);		//均值块滤波器
static void on_GaussianBlur(int, void*);			//高斯滤波器
static void on_MedianBlur(int, void*);			//中值滤波器
static void on_BilateralFilter(int, void*);			//双边滤波


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
	system("color 4F");

	// 载入原图
	g_srcImage = imread("1.jpg", 1);
	if (!g_srcImage.data) { printf("读取srcImage错误~！ \n"); return false; }

	//克隆原图到四个Mat类型中
	g_dstImage1 = g_srcImage.clone();
	g_dstImage2 = g_srcImage.clone();
	g_dstImage3 = g_srcImage.clone();
	g_dstImage4 = g_srcImage.clone();
	g_dstImage5 = g_srcImage.clone();

	//显示原图
	namedWindow("【<0>原图窗口】", 1);
	imshow("【<0>原图窗口】", g_srcImage);


	//=================【<1>方框滤波】=========================
	//创建窗口
	namedWindow("【<1>方框滤波】", 1);
	//创建轨迹条
	createTrackbar("内核值：", "【<1>方框滤波】", &g_nBoxFilterValue, 50, on_BoxFilter);
	on_MeanBlur(g_nBoxFilterValue, 0);
	imshow("【<1>方框滤波】", g_dstImage1);
	//=====================================================


	//=================【<2>均值滤波】==========================
	//创建窗口
	namedWindow("【<2>均值滤波】", 1);
	//创建轨迹条
	createTrackbar("内核值：", "【<2>均值滤波】", &g_nMeanBlurValue, 50, on_MeanBlur);
	on_MeanBlur(g_nMeanBlurValue, 0);
	//======================================================


	//=================【<3>高斯滤波】===========================
	//创建窗口
	namedWindow("【<3>高斯滤波】", 1);
	//创建轨迹条
	createTrackbar("内核值：", "【<3>高斯滤波】", &g_nGaussianBlurValue, 50, on_GaussianBlur);
	on_GaussianBlur(g_nGaussianBlurValue, 0);
	//=======================================================


	//=================【<4>中值滤波】===========================
	//创建窗口
	namedWindow("【<4>中值滤波】", 1);
	//创建轨迹条
	createTrackbar("参数值：", "【<4>中值滤波】", &g_nMedianBlurValue, 50, on_MedianBlur);
	on_MedianBlur(g_nMedianBlurValue, 0);
	//=======================================================


	//=================【<5>双边滤波】===========================
	//创建窗口
	namedWindow("【<5>双边滤波】", 1);
	//创建轨迹条
	createTrackbar("参数值：", "【<5>双边滤波】", &g_nBilateralFilterValue, 50, on_BilateralFilter);
	on_BilateralFilter(g_nBilateralFilterValue, 0);
	//=======================================================


	//输出一些帮助信息
	cout << endl << "\t运行成功，请调整滚动条观察图像效果~\n\n"
		<< "\t按下“q”键时，程序退出。\n";
	while (char(waitKey(1)) != 'q') {}

	return 0;
}

//-----------------------------【on_BoxFilter( )函数】------------------------------------
//		描述：方框滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_BoxFilter(int, void*)
{
	//方框滤波操作
	boxFilter(g_srcImage, g_dstImage1, -1, Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1));
	//显示窗口
	imshow("【<1>方框滤波】", g_dstImage1);
}

//-----------------------------【on_MeanBlur( )函数】------------------------------------
//		描述：均值滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_MeanBlur(int, void*)
{
	blur(g_srcImage, g_dstImage2, Size(g_nMeanBlurValue + 1, g_nMeanBlurValue + 1), Point(-1, -1));
	imshow("【<2>均值滤波】", g_dstImage2);

}

//-----------------------------【on_GaussianBlur( )函数】------------------------------------
//		描述：高斯滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_GaussianBlur(int, void*)
{
	GaussianBlur(g_srcImage, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
	imshow("【<3>高斯滤波】", g_dstImage3);
}


//-----------------------------【on_MedianBlur( )函数】------------------------------------
//		描述：中值滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_MedianBlur(int, void*)
{
	medianBlur(g_srcImage, g_dstImage4, g_nMedianBlurValue * 2 + 1);
	imshow("【<4>中值滤波】", g_dstImage4);
}


//-----------------------------【on_BilateralFilter( )函数】------------------------------------
//		描述：双边滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_BilateralFilter(int, void*)
{
	bilateralFilter(g_srcImage, g_dstImage5, g_nBilateralFilterValue, g_nBilateralFilterValue * 2, g_nBilateralFilterValue / 2);
	imshow("【<5>双边滤波】", g_dstImage5);
}
```

实验结果：
![avatar](\picture\24.非线性滤波.png)

### 6.3 形态学滤波
形态学(morphology）一词通常表示生物学的一个分支，该分支主要研究动植物的形态和结构。而我们图像处理中的形态学，往往指的是数学形态学。下面一起来了解数学形态学的概念。

数学形态学(Mathematical morphology）是一门建立在格论和拓扑学基础之上的图像分析学科，是数学形态学图像处理的基本理论。其基本的运算包括:二值腐蚀和膨胀、二值开闭运算、骨架抽取、极限腐蚀、击中击不中变换、形态学梯度、Top-hat变换、颗粒分析、流域变换、灰值腐蚀和膨胀、灰值开闭运算、灰值形态学梯度等。

简单来讲，形态学操作就是基于形状的一系列图像处理操作。OpenCV为进行图像的形态学变换提供了快捷、方便的函数。

### （1）：腐蚀与膨胀
最基本的形态学操作有两种，分别是:膨胀（dilate）与腐蚀（erode)。

膨胀与腐蚀能实现多种多样的功能，主要如下。
- 消除噪声;
- 分割(isolate）出独立的图像元素，在图像中连接(join）相邻的元素;
- 寻找图像中的明显的极大值区域或极小值区域;
- 求出图像的梯度。

#### 膨胀: dilate函数
膨胀（dilate）就是求局部最大值的操作。从数学角度来说，膨胀或者腐蚀操作就是将图像（或图像的一部分区域，称之为A）与核（称之为B）进行卷积。

膨胀就是求局部最大值的操作。核B与图形卷积，即计算核B覆盖的区域的像素点的最大值，并把这个最大值赋值给参考点指定的像素。这样就会使图像中的高亮区域逐渐增长，这就是膨胀操作的初衷。

dilate 函数使用像素邻域内的局部极大运算符来膨胀一张图片，从 src 输入，由dst输出。支持就地（in-place）操作。函数原型如下。
```C++
void dilate (
    InputArray src,
    outputArray dst,
    InputArray kernel,
    Point anchor=Point (-1,-1),
    int iterations=1,
    int borderType=BORDER_CONSTANT,
    const Scalar& bordervalue=morphologyDefaultBordervalue ());
```

#### 腐蚀：erode函数
膨胀和腐蚀(erode)是相反的一对操作，所以腐蚀就是求局部最小值的操作。

erode 函数使用像素邻域内的局部极小运算符来腐蚀一张图片，从 src 输入，由 dst 输出。支持就地(in-place)操作。函数原型，如下。
```C++
void erode (
    InputArray src,
    outputArray dst,
    InputArray kernel,
    Point anchor=Point (-1,-1 ),
    int iterations=1,
    int borderType=BORDER_CONSTANT,
    const Scalar& bordervalue=morphologyDefaultBordervalue()) ;
```

### 程序：腐蚀与膨胀
代码：
```C++
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage;//原始图和效果图
int g_nTrackbarNumer = 0;//0表示腐蚀erode, 1表示膨胀dilate
int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸


//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void Process();//膨胀和腐蚀的处理函数
void on_TrackbarNumChange(int, void*);//回调函数
void on_ElementSizeChange(int, void*);//回调函数

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
	//改变console字体颜色
	system("color 2F");

	//载入原图
	g_srcImage = imread("1.jpg");
	if (!g_srcImage.data) { printf("读取srcImage错误~！ \n"); return false; }

	//显示原始图
	namedWindow("【原始图】");
	imshow("【原始图】", g_srcImage);

	//进行初次腐蚀操作并显示效果图
	namedWindow("【效果图】");
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));
	erode(g_srcImage, g_dstImage, element);
	imshow("【效果图】", g_dstImage);

	//创建轨迹条
	createTrackbar("腐蚀/膨胀", "【效果图】", &g_nTrackbarNumer, 1, on_TrackbarNumChange);
	createTrackbar("内核尺寸", "【效果图】", &g_nStructElementSize, 21, on_ElementSizeChange);

	//输出一些帮助信息
	cout << endl << "\t运行成功，请调整滚动条观察图像效果~\n\n"
		<< "\t按下“q”键时，程序退出。\n";

	//轮询获取按键信息，若下q键，程序退出
	while (char(waitKey(1)) != 'q') {}

	return 0;
}

//-----------------------------【Process( )函数】------------------------------------
//		描述：进行自定义的腐蚀和膨胀操作
//-----------------------------------------------------------------------------------------
void Process()
{
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));

	//进行腐蚀或膨胀操作
	if (g_nTrackbarNumer == 0) {
		erode(g_srcImage, g_dstImage, element);
	}
	else {
		dilate(g_srcImage, g_dstImage, element);
	}

	//显示效果图
	imshow("【效果图】", g_dstImage);
}


//-----------------------------【on_TrackbarNumChange( )函数】------------------------------------
//		描述：腐蚀和膨胀之间切换开关的回调函数
//-----------------------------------------------------------------------------------------------------
void on_TrackbarNumChange(int, void*)
{
	//腐蚀和膨胀之间效果已经切换，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
	Process();
}


//-----------------------------【on_ElementSizeChange( )函数】-------------------------------------
//		描述：腐蚀和膨胀操作内核改变时的回调函数
//-----------------------------------------------------------------------------------------------------
void on_ElementSizeChange(int, void*)
{
	//内核尺寸已改变，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
	Process();
}

```

实验结果：
![avatar](\picture\25.形态学滤波1.png)
![avatar](\picture\25.形态学滤波2.png)

### （2）：开闭梯度

#### 开运算
开运算(Opening Operation)，其实就是先腐蚀后膨胀的过程。其数学表达式如下: dst=open (src,element)=dilate(erode (src,element)) 

开运算可以用来消除小物体，在纤细点处分离物体，并且在平滑较大物体的边界的同时不明显改变其面积。

#### 闭运算
先膨胀后腐蚀的过程称为闭运算(Closing Operation)，其数学表达式如下: dst=clese(src,element)= erode (dilate(src,element))

闭运算能够排除小型黑洞（黑色区域)。

#### 形态学梯度
形态学梯度(Morphological Gradient）是膨胀图与腐蚀图之差，数学表达式如下: dst=morph-grad(src,element)= dilate(src,element)- erode (src,element) 

对二值图像进行这一操作可以将团块（blob）的边缘突出出来。我们可以用形态学梯度来保留物体的边缘轮廓。

#### 顶帽
顶帽运算（Top Hat）又常常被译为”礼帽“运算，是原图像与上文刚刚介绍的“开运算”的结果图之差，数学表达式如下: dst-tophat (src,element ) -src-open(arc,element)

顶帽运算往往用来分离比邻近点亮一些的斑块。在一幅图像具有大幅的背景，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取。

#### 核心API函数：morphologyEx()
morphologyEx函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换，如开闭运算、形态学梯度、“顶帽”、“黑帽”等。
```C++
void morphologyEx(
    InputArray src,
    outputArray dst,
    int op,
    InputArraykernel,
    Pointanchor=Point (-1,-1),
    intiterations=1,
    intborderType=BORDER_CONSTANT,
    constScalar& bordervalue=morphologyDefaultBordervalue() );
```

### 程序：形态学滤波
代码：
```C++
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


//-----------------------------------【全局变量声明部分】-----------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage;//原始图和效果图
int g_nElementShape = MORPH_RECT;//元素结构的形状

//变量接收的TrackBar位置参数
int g_nMaxIterationNum = 10;
int g_nOpenCloseNum = 0;
int g_nErodeDilateNum = 0;
int g_nTopBlackHatNum = 0;



//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------
static void on_OpenClose(int, void*);//回调函数
static void on_ErodeDilate(int, void*);//回调函数
static void on_TopBlackHat(int, void*);//回调函数


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
	//改变console字体颜色
	system("color 2F");

	//载入原图
	g_srcImage = imread("1.jpg");
	if (!g_srcImage.data) { printf("Oh，no，读取srcImage错误~！ \n"); return false; }

	//显示原始图
	namedWindow("【原始图】");
	imshow("【原始图】", g_srcImage);

	//创建三个窗口
	namedWindow("【开运算/闭运算】", 1);
	namedWindow("【腐蚀/膨胀】", 1);
	namedWindow("【顶帽/黑帽】", 1);

	//参数赋值
	g_nOpenCloseNum = 9;
	g_nErodeDilateNum = 9;
	g_nTopBlackHatNum = 2;

	//分别为三个窗口创建滚动条
	createTrackbar("迭代值", "【开运算/闭运算】", &g_nOpenCloseNum, g_nMaxIterationNum * 2 + 1, on_OpenClose);
	createTrackbar("迭代值", "【腐蚀/膨胀】", &g_nErodeDilateNum, g_nMaxIterationNum * 2 + 1, on_ErodeDilate);
	createTrackbar("迭代值", "【顶帽/黑帽】", &g_nTopBlackHatNum, g_nMaxIterationNum * 2 + 1, on_TopBlackHat);

	//轮询获取按键信息
	while (1)
	{
		int c;

		//执行回调函数
		on_OpenClose(g_nOpenCloseNum, 0);
		on_ErodeDilate(g_nErodeDilateNum, 0);
		on_TopBlackHat(g_nTopBlackHatNum, 0);

		//获取按键
		c = waitKey(0);

		//按下键盘按键Q或者ESC，程序退出
		if ((char)c == 'q' || (char)c == 27)
			break;
		//按下键盘按键1，使用椭圆(Elliptic)结构元素结构元素MORPH_ELLIPSE
		if ((char)c == 49)//键盘按键1的ASII码为49
			g_nElementShape = MORPH_ELLIPSE;
		//按下键盘按键2，使用矩形(Rectangle)结构元素MORPH_RECT
		else if ((char)c == 50)//键盘按键2的ASII码为50
			g_nElementShape = MORPH_RECT;
		//按下键盘按键3，使用十字形(Cross-shaped)结构元素MORPH_CROSS
		else if ((char)c == 51)//键盘按键3的ASII码为51
			g_nElementShape = MORPH_CROSS;
		//按下键盘按键space，在矩形、椭圆、十字形结构元素中循环
		else if ((char)c == ' ')
			g_nElementShape = (g_nElementShape + 1) % 3;
	}

	return 0;
}


//-----------------------------------【on_OpenClose( )函数】----------------------------------
//		描述：【开运算/闭运算】窗口的回调函数
//-----------------------------------------------------------------------------------------------
static void on_OpenClose(int, void*)
{
	//偏移量的定义
	int offset = g_nOpenCloseNum - g_nMaxIterationNum;//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//进行操作
	if (offset < 0)
		//此句代码的OpenCV2版为：
		//morphologyEx(g_srcImage, g_dstImage, CV_MOP_OPEN, element);
		//此句代码的OpenCV3版为:
		morphologyEx(g_srcImage, g_dstImage, MORPH_OPEN, element);
	else
		//此句代码的OpenCV2版为：
		//morphologyEx(g_srcImage, g_dstImage, CV_MOP_CLOSE, element);
		//此句代码的OpenCV3版为:
		morphologyEx(g_srcImage, g_dstImage, MORPH_CLOSE, element);



	//显示图像
	imshow("【开运算/闭运算】", g_dstImage);
}


//-----------------------------------【on_ErodeDilate( )函数】----------------------------------
//		描述：【腐蚀/膨胀】窗口的回调函数
//-----------------------------------------------------------------------------------------------
static void on_ErodeDilate(int, void*)
{
	//偏移量的定义
	int offset = g_nErodeDilateNum - g_nMaxIterationNum;	//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//进行操作
	if (offset < 0)
		erode(g_srcImage, g_dstImage, element);
	else
		dilate(g_srcImage, g_dstImage, element);
	//显示图像
	imshow("【腐蚀/膨胀】", g_dstImage);
}


//-----------------------------------【on_TopBlackHat( )函数】--------------------------------
//		描述：【顶帽运算/黑帽运算】窗口的回调函数
//----------------------------------------------------------------------------------------------
static void on_TopBlackHat(int, void*)
{
	//偏移量的定义
	int offset = g_nTopBlackHatNum - g_nMaxIterationNum;//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//进行操作
	if (offset < 0)
		morphologyEx(g_srcImage, g_dstImage, MORPH_TOPHAT, element);
	else
		morphologyEx(g_srcImage, g_dstImage, MORPH_BLACKHAT, element);
	//显示图像
	imshow("【顶帽/黑帽】", g_dstImage);
}
```

实验结果：
![avatar](\picture\26.形态学滤波.png)

### 6.4 漫水填充

#### 定义
漫水填充法是一种用特定的颜色填充连通区域，通过设置可连通像素的上下限以及连通方式来达到不同的填充效果的方法。漫水填充经常被用来标记或分离图像的一部分，以便对其进行进一步处理或分析，也可以用来从输入图像获取掩码区域，掩码会加速处理过程，或只处理掩码指定的像素点，操作的结果总是某个连续的区域。

#### 基本思想
所谓漫水填充，简单来说，就是自动选中了和种子点相连的区域，接着将该区域替换成指定的颜色，这是个非常有用的功能，经常用来标记或者分离图像的一部分进行处理或分析。漫水填充也可以用来从输入图像获取掩码区域，掩码会加速处理过程，或者只处理掩码指定的像素点。

以此填充算法为基础，类似 PhotoShop的魔术棒选择工具就很容易实现了。漫水填充（FloodFill)是查找和种子点连通的颜色相同的点，魔术棒选择工具则是查找和种子点连通的颜色相近的点，把和初始种子像素颜色相近的点压进栈做为新种子。

在OpenCV中，漫水填充是填充算法中最通用的方法

#### 实现算法：floodFill函数
漫水填充算法由 floodFill 函数实现，其作用是用我们指定的颜色从种子点开始填充一个连接域。连通性由像素值的接近程度来衡量。OpenCV2.X 有两个C++重写版本的 floodFill，具体如下。

第一个版本的floodFill:
```C++
int floodFill(InputOutputArray image，Point seedPoint，Scalar newVal, Rect* rect=0，Scalar loDiff=Scalar (),Scalar upDiff=Scalar(), intflags=4)
```
第二个版本的floodFill:
```C++
int floodFill(InputOutputArray image，InputOutputArray mask,PointseedPoint,Scalar newVal，Rect* rect=0，Scalar loDiff=Scalar()，ScalarupDiff=Scalar(), int flags=4 )
```

### 程序：漫水填充
代码：
```C++
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【全局变量声明部分】--------------------------------------  
//      描述：全局变量声明  
//-----------------------------------------------------------------------------------------------  
Mat g_srcImage, g_dstImage, g_grayImage, g_maskImage;//定义原始图、目标图、灰度图、掩模图
int g_nFillMode = 1;//漫水填充的模式
int g_nLowDifference = 20, g_nUpDifference = 20;//负差最大值、正差最大值
int g_nConnectivity = 4;//表示floodFill函数标识符低八位的连通值
int g_bIsColor = true;//是否为彩色图的标识符布尔值
bool g_bUseMask = false;//是否显示掩膜窗口的布尔值
int g_nNewMaskVal = 255;//新的重新绘制的像素值

//-----------------------------------【onMouse( )函数】--------------------------------------  
//      描述：鼠标消息onMouse回调函数
//---------------------------------------------------------------------------------------------
static void onMouse(int event, int x, int y, int, void*)
{
	// 若鼠标左键没有按下，便返回
	//此句代码的OpenCV2版为：
	//if( event != CV_EVENT_LBUTTONDOWN )
	//此句代码的OpenCV3版为：
	if (event != EVENT_LBUTTONDOWN)
		return;

	//-------------------【<1>调用floodFill函数之前的参数准备部分】---------------
	Point seed = Point(x, y);
	int LowDifference = g_nFillMode == 0 ? 0 : g_nLowDifference;//空范围的漫水填充，此值设为0，否则设为全局的g_nLowDifference
	int UpDifference = g_nFillMode == 0 ? 0 : g_nUpDifference;//空范围的漫水填充，此值设为0，否则设为全局的g_nUpDifference

	//标识符的0~7位为g_nConnectivity，8~15位为g_nNewMaskVal左移8位的值，16~23位为CV_FLOODFILL_FIXED_RANGE或者0。
	//此句代码的OpenCV2版为：
	//int flags = g_nConnectivity + (g_nNewMaskVal << 8) +(g_nFillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
	//此句代码的OpenCV3版为：
	int flags = g_nConnectivity + (g_nNewMaskVal << 8) + (g_nFillMode == 1 ? FLOODFILL_FIXED_RANGE : 0);

	//随机生成bgr值
	int b = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	int g = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	int r = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	Rect ccomp;//定义重绘区域的最小边界矩形区域

	Scalar newVal = g_bIsColor ? Scalar(b, g, r) : Scalar(r * 0.299 + g * 0.587 + b * 0.114);//在重绘区域像素的新值，若是彩色图模式，取Scalar(b, g, r)；若是灰度图模式，取Scalar(r*0.299 + g*0.587 + b*0.114)

	Mat dst = g_bIsColor ? g_dstImage : g_grayImage;//目标图的赋值
	int area;

	//--------------------【<2>正式调用floodFill函数】-----------------------------
	if (g_bUseMask)
	{
		//此句代码的OpenCV2版为：
		//threshold(g_maskImage, g_maskImage, 1, 128, CV_THRESH_BINARY);
		//此句代码的OpenCV3版为：
		threshold(g_maskImage, g_maskImage, 1, 128, THRESH_BINARY);
		area = floodFill(dst, g_maskImage, seed, newVal, &ccomp, Scalar(LowDifference, LowDifference, LowDifference),
			Scalar(UpDifference, UpDifference, UpDifference), flags);
		imshow("mask", g_maskImage);
	}
	else
	{
		area = floodFill(dst, seed, newVal, &ccomp, Scalar(LowDifference, LowDifference, LowDifference),
			Scalar(UpDifference, UpDifference, UpDifference), flags);
	}

	imshow("效果图", dst);
	cout << area << " 个像素被重绘\n";
}


//-----------------------------------【main( )函数】--------------------------------------------  
//      描述：控制台应用程序的入口函数，我们的程序从这里开始  
//-----------------------------------------------------------------------------------------------  
int main(int argc, char** argv)
{
	//改变console字体颜色  
	system("color 2F");

	//载入原图
	g_srcImage = imread("1.jpg", 1);

	if (!g_srcImage.data) { printf("读取图片image0错误~！ \n"); return false; }

	g_srcImage.copyTo(g_dstImage);//拷贝源图到目标图
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);//转换三通道的image0到灰度图
	g_maskImage.create(g_srcImage.rows + 2, g_srcImage.cols + 2, CV_8UC1);//利用image0的尺寸来初始化掩膜mask

	//此句代码的OpenCV2版为：
	//namedWindow( "效果图",CV_WINDOW_AUTOSIZE );
	//此句代码的OpenCV2版为：
	namedWindow("效果图", WINDOW_AUTOSIZE);


	//创建Trackbar
	createTrackbar("负差最大值", "效果图", &g_nLowDifference, 255, 0);
	createTrackbar("正差最大值", "效果图", &g_nUpDifference, 255, 0);

	//鼠标回调函数
	setMouseCallback("效果图", onMouse, 0);

	//循环轮询按键
	while (1)
	{
		//先显示效果图
		imshow("效果图", g_bIsColor ? g_dstImage : g_grayImage);

		//获取键盘按键
		int c = waitKey(0);
		//判断ESC是否按下，若按下便退出
		if ((c & 255) == 27)
		{
			cout << "程序退出...........\n";
			break;
		}

		//根据按键的不同，进行各种操作
		switch ((char)c)
		{
			//如果键盘“1”被按下，效果图在在灰度图，彩色图之间互换
		case '1':
			if (g_bIsColor)//若原来为彩色，转为灰度图，并且将掩膜mask所有元素设置为0
			{
				cout << "键盘“1”被按下，切换彩色/灰度模式，当前操作为将【彩色模式】切换为【灰度模式】\n";
				cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
				g_maskImage = Scalar::all(0);	//将mask所有元素设置为0
				g_bIsColor = false;	//将标识符置为false，表示当前图像不为彩色，而是灰度
			}
			else//若原来为灰度图，便将原来的彩图image0再次拷贝给image，并且将掩膜mask所有元素设置为0
			{
				cout << "键盘“1”被按下，切换彩色/灰度模式，当前操作为将【彩色模式】切换为【灰度模式】\n";
				g_srcImage.copyTo(g_dstImage);
				g_maskImage = Scalar::all(0);
				g_bIsColor = true;//将标识符置为true，表示当前图像模式为彩色
			}
			break;
			//如果键盘按键“2”被按下，显示/隐藏掩膜窗口
		case '2':
			if (g_bUseMask)
			{
				destroyWindow("mask");
				g_bUseMask = false;
			}
			else
			{
				namedWindow("mask", 0);
				g_maskImage = Scalar::all(0);
				imshow("mask", g_maskImage);
				g_bUseMask = true;
			}
			break;
			//如果键盘按键“3”被按下，恢复原始图像
		case '3':
			cout << "按键“3”被按下，恢复原始图像\n";
			g_srcImage.copyTo(g_dstImage);
			cvtColor(g_dstImage, g_grayImage, COLOR_BGR2GRAY);
			g_maskImage = Scalar::all(0);
			break;
			//如果键盘按键“4”被按下，使用空范围的漫水填充
		case '4':
			cout << "按键“4”被按下，使用空范围的漫水填充\n";
			g_nFillMode = 0;
			break;
			//如果键盘按键“5”被按下，使用渐变、固定范围的漫水填充
		case '5':
			cout << "按键“5”被按下，使用渐变、固定范围的漫水填充\n";
			g_nFillMode = 1;
			break;
			//如果键盘按键“6”被按下，使用渐变、浮动范围的漫水填充
		case '6':
			cout << "按键“6”被按下，使用渐变、浮动范围的漫水填充\n";
			g_nFillMode = 2;
			break;
			//如果键盘按键“7”被按下，操作标志符的低八位使用4位的连接模式
		case '7':
			cout << "按键“7”被按下，操作标志符的低八位使用4位的连接模式\n";
			g_nConnectivity = 4;
			break;
			//如果键盘按键“8”被按下，操作标志符的低八位使用8位的连接模式
		case '8':
			cout << "按键“8”被按下，操作标志符的低八位使用8位的连接模式\n";
			g_nConnectivity = 8;
			break;
		}
	}

	return 0;
}

```

实验结果：
![avatar](\picture\27.漫水填充.png)

### 6.5 图像金字塔与图片尺寸缩放

#### 图像金字塔
图像金字塔是图像中多尺度表达的一种，最主要用于图像的分割，是一种以多分辨率来解释图像的有效但概念简单的结构。

金字塔的底部是待处理图像的高分辨率表示，而顶部是低分辨率的近似。我们将一层一层的图像比喻成金字塔，层级越高，则图像越小，分辨率越低。

一般情况下有两种类型的图像金字塔常常出现在文献和以及实际运用中。它们分别是:
- 高斯金字塔(Gaussianpyramid)———用来向下采样，主要的图像金字塔。
- 拉普拉斯金字塔(Laplacianpyramid)——用来从金字塔低层图像重建上层未采样图像，在数字图像处理中也即是预测残差，可以对图像进行最大程度的还原，配合高斯金字塔一起使用。

两者的简要区别在于：高斯金字塔用来向下降采样图像，而拉普拉斯金字塔则用来从金字塔底层图像中向上采样，重建一个图像。

#### 向上采样: pyrUp()函数
pyrUp()函数的作用是向上采样并模糊一张图像，说白了就是放大一张图片。
```C++
void pyrup(InputArray src，outputArraydst,const size&dstsize=size() , int borderType=BORDER_DEFAULT )
```

#### 向下采样：pyrDown()函数
pyrDown()函数的作用是向下采样并模糊一张图片，说白了就是缩小一张图片。
```C++
void pyrDown (InputArray src,outputArray dst,const size&dstsize=size ( ) , int borderType=BORDER_DEFAULT)
```

#### 尺寸调整：resize()函数
resize()为 OpenCV 中专门用来调整图像大小的函数。

此函数将源图像精确地转换为指定尺寸的目标图像。如果源图像中设置了ROI (Region Of Interest ,感兴趣区域)，那么resize()函数会对源图像的ROI区域进行调整图像尺寸的操作,来输出到目标图像中。若目标图像中已经设置了ROI区域，不难理解resize()将会对源图像进行尺寸调整并填充到目标图像的ROI中。

其尺寸和类型可以由src、dsize、fx和fy这几个参数来确定。函数原型:
```C++
void resize(InputArray src,outputArray dst,size dsize,double fx=0,double fy=0, int interpolation=INTER_LINEAR )
```

### 程序：图像金字塔与缩放
代码：
```C++
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


//-----------------------------------【宏定义部分】--------------------------------------------
//	描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME "【程序窗口】"		//为窗口标题定义的宏


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage, g_tmpImage;

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
	//改变console字体颜色
	system("color 2F");

	//载入原图
	g_srcImage = imread("1.jpg");//工程目录下需要有一张名为1.jpg的测试图像，且其尺寸需被2的N次方整除，N为可以缩放的次数
	if (!g_srcImage.data) { printf("Oh，no，读取srcImage错误~！ \n"); return false; }

	// 创建显示窗口
	namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME, g_srcImage);

	//参数赋值
	g_tmpImage = g_srcImage;
	g_dstImage = g_tmpImage;

	int key = 0;

	//轮询获取按键信息
	while (1)
	{
		key = waitKey(9);//读取键值到key变量中

		//根据key变量的值，进行不同的操作
		switch (key)
		{
			//======================【程序退出相关键值处理】=======================  
		case 27://按键ESC
			return 0;
			break;

		case 'q'://按键Q
			return 0;
			break;

			//======================【图片放大相关键值处理】=======================  
		case 'a'://按键A按下，调用pyrUp函数
			pyrUp(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">检测到按键【A】被按下，开始进行基于【pyrUp】函数的图片放大：图片尺寸×2 \n");
			break;

		case 'w'://按键W按下，调用resize函数
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">检测到按键【W】被按下，开始进行基于【resize】函数的图片放大：图片尺寸×2 \n");
			break;

		case '1'://按键1按下，调用resize函数
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">检测到按键【1】被按下，开始进行基于【resize】函数的图片放大：图片尺寸×2 \n");
			break;

		case '3': //按键3按下，调用pyrUp函数
			pyrUp(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">检测到按键【3】被按下，开始进行基于【pyrUp】函数的图片放大：图片尺寸×2 \n");
			break;
			//======================【图片缩小相关键值处理】=======================  
		case 'd': //按键D按下，调用pyrDown函数
			pyrDown(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			printf(">检测到按键【D】被按下，开始进行基于【pyrDown】函数的图片缩小：图片尺寸/2\n");
			break;

		case  's': //按键S按下，调用resize函数
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			printf(">检测到按键【S】被按下，开始进行基于【resize】函数的图片缩小：图片尺寸/2\n");
			break;

		case '2'://按键2按下，调用resize函数
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			printf(">检测到按键【2】被按下，开始进行基于【resize】函数的图片缩小：图片尺寸/2\n");
			break;

		case '4': //按键4按下，调用pyrDown函数
			pyrDown(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			printf(">检测到按键【4】被按下，开始进行基于【pyrDown】函数的图片缩小：图片尺寸/2\n");
			break;
		}

		//经过操作后，显示变化后的图
		imshow(WINDOW_NAME, g_dstImage);

		//将g_dstImage赋给g_tmpImage，方便下一次循环
		g_tmpImage = g_dstImage;
	}

	return 0;
}
```

实验结果：
![avatar](\picture\28.图像缩放.png)

### 6.6 阈值化

#### 简介
阈值可以被视作最简单的图像分割方法。这样的图像分割方法基于图像中物体与背景之间的灰度差异，而且此分割属于像素级的分割。为了从一副图像中提取出我们需要的部分，应该用图像中的每一个像素点的灰度值与选取的阈值进行比较，并作出相应的判断。

阈值的选取依赖于具体的问题。即物体在不同的图像中有可能会有不同的灰度值。

一旦找到了需要分割的物体的像素点，可以对这些像素点设定一些特定的值来表示。

#### 固定阈值操作：Threshold()函数
函数Threshold()对单通道数组应用固定阈值操作。该函数的典型应用是对灰度图像进行阈值操作得到二值图像，( compare()函数也可以达到此目的）或者是去掉噪声，例如过滤很小或很大象素值的图像点。函数原型如下。
```C++
double threshold (InputArray src,outputArray dst, double thresh,double maxval, int type)
```

#### 自适应阈值操作: adaptiveThreshold()函数
adaptiveThreshold()函数的作用是对矩阵采用自适应阈值操作，支持就地操作。函数原型如下。
```C++
 void adaptiveThreshold(InputArray src, OutputArray dst,doublemaxValue, int adaptiveMethod, int thresholdType, int blockSize, doublec)
```

### 程序：阈值操作
代码：
```C++
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//-----------------------------------【宏定义部分】-------------------------------------------- 
//		描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME "【程序窗口】"        //为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
int g_nThresholdValue = 100;
int g_nThresholdType = 3;
Mat g_srcImage, g_grayImage, g_dstImage;

//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
void on_Threshold(int, void*);//回调函数


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	//【0】改变console字体颜色
	system("color 1F");

	//【1】读入源图片
	g_srcImage = imread("1.jpg");
	if (!g_srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }
	imshow("原始图", g_srcImage);

	//【2】存留一份原图的灰度图
	cvtColor(g_srcImage, g_grayImage, COLOR_RGB2GRAY);

	//【3】创建窗口并显示原始图
	namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);

	//【4】创建滑动条来控制阈值
	createTrackbar("模式",
		WINDOW_NAME, &g_nThresholdType,
		4, on_Threshold);

	createTrackbar("参数值",
		WINDOW_NAME, &g_nThresholdValue,
		255, on_Threshold);

	//【5】初始化自定义的阈值回调函数
	on_Threshold(0, 0);

	// 【6】轮询等待用户按键，如果ESC键按下则退出程序
	while (1)
	{
		int key;
		key = waitKey(20);
		if ((char)key == 27) { break; }
	}

}

//-----------------------------------【on_Threshold( )函数】------------------------------------
//		描述：自定义的阈值回调函数
//-----------------------------------------------------------------------------------------------
void on_Threshold(int, void*)
{
	//调用阈值函数
	threshold(g_grayImage, g_dstImage, g_nThresholdValue, 255, g_nThresholdType);

	//更新效果图
	imshow(WINDOW_NAME, g_dstImage);
}
```

实验结果：
![avatar](\picture\29.阈值操作0.png)
![avatar](\picture\29.阈值操作1.png)
![avatar](\picture\29.阈值操作2.png)
![avatar](\picture\29.阈值操作3.png)
![avatar](\picture\29.阈值操作4.png)

## 七、图像变换

### 7.1 边缘检测

#### 步骤
- 【第一步】滤波
  
  边缘检测的算法主要是基于图像强度的一阶和二阶导数，但导数通常对噪声很敏感，因此必须采用滤波器来改善与噪声有关的边缘检测器的性能。常见的滤波方法主要有高斯滤波，即采用离散化的高斯函数产生一组归一化的高斯核，然后基于高斯核函数对图像灰度矩阵的每一点进行加权求和。
- 【第二步】增强
  
  增强边缘的基础是确定图像各点邻域强度的变化值。增强算法可以将图像灰度点邻域强度值有显著变化的点凸显出来。在具体编程实现时，可通过计算梯度幅值来确定。
- 【第三步】检测
  
  经过增强的图像，往往邻域中有很多点的梯度值比较大，而在特定的应用中，这些点并不是要找的边缘点，所以应该采用某种方法来对这些点进行取舍。实际工程中，常用的方法是通过阈值化方法来检测。

####  Canny边缘检测：Canny()函数
Canny函数利用Canny算子来进行图像的边缘检测操作。
```C++
void Canny(InputArray image,OutputArray edges,double threshold1,double threshold2, int aperturesize=3,bool L2gradient=false )
```

#### 使用Sobel算子: Sobel()函数
Sobel函数使用扩展的Sobel算子，来计算一阶、二阶、三阶或混合图像差分。
```C++
void sobel (
    InputArray src,
    OutputArray dst,
    int ddepth,
    int dx,
    int dy,
    int ksize=3,
    double scale=1,
    double delta=0,
    int borderType=BORDER_DEFAULT );
```

#### 计算拉普拉斯变换: Laplacian()函数
Laplacian函数可以计算出图像经过拉普拉斯变换后的结果。
```C++
void Laplacian(InputArray src,outputArray dst,int ddepth,intksize=l, double scale=l, double deita=0,intborderType=BORDER__DEFAULT ) ;
```

#### 计算图像差分：Scharr()函数
使用 Scharr滤波器运算符计算x或y方向的图像差分。其实它的参数变量和Sobel基本上是一样的，除了没有ksize核的大小。
```C++
void scharr(
    InputArray src, //源图
    OutputArray dst, //目标图
    int ddepth, //图像深度
    int dx, //x方向上的差分阶数
    int dy, //y方向上的差分阶数
    double scale=1, //缩放因子
    double delta=0, //delta值
    intborderType=BORDER_DEFAULT )//边界模式
```

### 程序：边缘检测
代码：
```C++
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
//原图，原图的灰度版，目标图
Mat g_srcImage, g_srcGrayImage, g_dstImage;

//Canny边缘检测相关变量
Mat g_cannyDetectedEdges;
int g_cannyLowThreshold = 1;//TrackBar位置参数  

//Sobel边缘检测相关变量
Mat g_sobelGradient_X, g_sobelGradient_Y;
Mat g_sobelAbsGradient_X, g_sobelAbsGradient_Y;
int g_sobelKernelSize = 1;//TrackBar位置参数  

//Scharr滤波器相关变量
Mat g_scharrGradient_X, g_scharrGradient_Y;
Mat g_scharrAbsGradient_X, g_scharrAbsGradient_Y;


//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------
static void on_Canny(int, void*);//Canny边缘检测窗口滚动条的回调函数
static void on_Sobel(int, void*);//Sobel边缘检测窗口滚动条的回调函数
void Scharr();//封装了Scharr边缘检测相关代码的函数


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//改变console字体颜色
	system("color 2F");

	//载入原图
	g_srcImage = imread("1.jpg");
	if (!g_srcImage.data) { printf("Oh，no，读取srcImage错误~！ \n"); return false; }

	//显示原始图
	namedWindow("【原始图】");
	imshow("【原始图】", g_srcImage);

	// 创建与src同类型和大小的矩阵(dst)
	g_dstImage.create(g_srcImage.size(), g_srcImage.type());

	// 将原图像转换为灰度图像
	cvtColor(g_srcImage, g_srcGrayImage, COLOR_BGR2GRAY);

	// 创建显示窗口
	namedWindow("【效果图】Canny边缘检测", WINDOW_AUTOSIZE);
	namedWindow("【效果图】Sobel边缘检测", WINDOW_AUTOSIZE);

	// 创建trackbar
	createTrackbar("参数值：", "【效果图】Canny边缘检测", &g_cannyLowThreshold, 120, on_Canny);
	createTrackbar("参数值：", "【效果图】Sobel边缘检测", &g_sobelKernelSize, 3, on_Sobel);

	// 调用回调函数
	on_Canny(0, 0);
	on_Sobel(0, 0);

	//调用封装了Scharr边缘检测代码的函数
	Scharr();

	//轮询获取按键信息，若按下Q，程序退出
	while ((char(waitKey(1)) != 'q')) {}

	return 0;
}

//-----------------------------------【on_Canny( )函数】----------------------------------
//		描述：Canny边缘检测窗口滚动条的回调函数
//-----------------------------------------------------------------------------------------------
void on_Canny(int, void*)
{
	// 先使用 3x3内核来降噪
	blur(g_srcGrayImage, g_cannyDetectedEdges, Size(3, 3));

	// 运行我们的Canny算子
	Canny(g_cannyDetectedEdges, g_cannyDetectedEdges, g_cannyLowThreshold, g_cannyLowThreshold * 3, 3);

	//先将g_dstImage内的所有元素设置为0 
	g_dstImage = Scalar::all(0);

	//使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中
	g_srcImage.copyTo(g_dstImage, g_cannyDetectedEdges);

	//显示效果图
	imshow("【效果图】Canny边缘检测", g_dstImage);
}



//-----------------------------------【on_Sobel( )函数】----------------------------------
//		描述：Sobel边缘检测窗口滚动条的回调函数
//-----------------------------------------------------------------------------------------
void on_Sobel(int, void*)
{
	// 求 X方向梯度
	Sobel(g_srcImage, g_sobelGradient_X, CV_16S, 1, 0, (2 * g_sobelKernelSize + 1), 1, 1, BORDER_DEFAULT);
	convertScaleAbs(g_sobelGradient_X, g_sobelAbsGradient_X);//计算绝对值，并将结果转换成8位

	// 求Y方向梯度
	Sobel(g_srcImage, g_sobelGradient_Y, CV_16S, 0, 1, (2 * g_sobelKernelSize + 1), 1, 1, BORDER_DEFAULT);
	convertScaleAbs(g_sobelGradient_Y, g_sobelAbsGradient_Y);//计算绝对值，并将结果转换成8位

	// 合并梯度
	addWeighted(g_sobelAbsGradient_X, 0.5, g_sobelAbsGradient_Y, 0.5, 0, g_dstImage);

	//显示效果图
	imshow("【效果图】Sobel边缘检测", g_dstImage);

}


//-----------------------------------【Scharr( )函数】----------------------------------
//		描述：封装了Scharr边缘检测相关代码的函数
//-----------------------------------------------------------------------------------------
void Scharr()
{
	// 求 X方向梯度
	Scharr(g_srcImage, g_scharrGradient_X, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(g_scharrGradient_X, g_scharrAbsGradient_X);//计算绝对值，并将结果转换成8位

	// 求Y方向梯度
	Scharr(g_srcImage, g_scharrGradient_Y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(g_scharrGradient_Y, g_scharrAbsGradient_Y);//计算绝对值，并将结果转换成8位

	// 合并梯度
	addWeighted(g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0, g_dstImage);

	//显示效果图
	imshow("【效果图】Scharr滤波器", g_dstImage);
}
```

实验结果：
![avatar](\picture\30.边缘检测.png)

### 7.2 霍夫变换

#### 概述
霍夫变换（Hough Transform）是图像处理中的一种特征提取技术，该过程在一个参数空间中通过计算累计结果的局部最大值得到一个符合该特定形状的集合作为霍夫变换结果。

霍夫变换在 OpenCV中分为霍夫线变换和霍夫圆变换两种。

OpenCV支持三种不同的霍夫线变换，它们分别是:标准霍夫变换(StandardHough Transform，SHT)、多尺度霍夫变换(Multi-Scale Hough Transform，MSHT)和累计概率霍夫变换（Progressive Probabilistic Hough Transform，PPHT)。

#### 标准霍夫变换：HoughLines()函数
此函数可以找出采用标准霍夫变换的二值图像线条。在 OpenCV 中，我们可以用其来调用标准霍夫变换SHT和多尺度霍夫变换MSHT的OpenCV内建算法。
```C++
void HoughLines(InputArray image，OutputArray lines，double rho,double theta,int threshold，double srn=0，double stn=0 )
```

#### 累计概率霍夫变换：HoughLinesP()函数
此函数在HoughLines的基础上，在末尾加了一个代表Probabilistic（概率)的P，表明它可以采用累计概率霍夫变换（PPHT）来找出二值图像中的直线。
```C++
void HoughLinesP (InputArray image, outputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )
```

#### 原理
霍夫梯度法的原理是这样的：
  1) 首先对图像应用边缘检测，比如用canny边缘检测。
  2) 然后，对边缘图像中的每一个非零点，考虑其局部梯度，即用Sobel()函数计算x和y方向的Sobel一阶导数得到梯度。
  3) 利用得到的梯度，由斜率指定的直线上的每一个点都在累加器中被累加，这里的斜率是从一个指定的最小值到指定的最大值的距离。
  4) 同时，标记边缘图像中每一个非0像素的位置。
  5) 然后从二维累加器中这些点中选择候选的中心，这些中心都大于给定阈值并且大于其所有近邻。这些候选的中心按照累加值降序排列，以便于最支持像素的中心首先出现。
  6) 接下来对每一个中心，考虑所有的非0像素。
  7) 这些像素按照其与中心的距离排序。从到最大半径的最小距离算起，选择非0像素最支持的一条半径。
  8) 如果一个中心收到边缘图像非0像素最充分的支持，并且到前期被选择的中心有足够的距离，那么它就会被保留下来。

这个实现可以使算法执行起来更高效，或许更加重要的是，能够帮助解决三维累加器中会产生许多噪声并且使得结果不稳定的稀疏分布问题。

#### 缺点
1) 在霍夫梯度法中，我们使用Sobel导数来计算局部梯度，那么随之而来的假设是，它可以视作等同于一条局部切线，这并不是一个数值稳定的做法。在大多数情况下，这样做会得到正确的结果，但或许会在输出中产生一些噪声。
2) 在边缘图像中的整个非0像素集被看做每个中心的候选部分。因此，如果把累加器的阈值设置偏低，算法将要消耗比较长的时间。此外，因为每一个中心只选择一个圆,如果有同心圆，就只能选择其中的一个。
3) 因为中心是按照其关联的累加器值的升序排列的，并且如果新的中心过于接近之前已经接受的中心的话，就不会被保留下来。且当有许多同心圆或者是近似的同心圆时，霍夫梯度法的倾向是保留最大的一个圆。

####　霍夫圆变换：HoughCircles()函数
HoughCircles　函数可以利用霍夫变换算法检测出灰度图中的圆。它相比之前的　HoughLines　和　HoughLinesP，比较明显的一个区别是不需要源图是二值的，而　HoughLines　和 HoughLinesP　都需要源图为二值图像。
```C++
void Houghcircles(InputArray image, outputArray circles,int method,double dp,double minDist, double paraml=100 , double param2=100,intminRadius=0,int maxRadius=0 )
```

### 程序：霍夫变换
代码：
```C++
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage, g_midImage;//原始图、中间图和效果图
vector<Vec4i> g_lines;//定义一个矢量结构g_lines用于存放得到的线段矢量集合
//变量接收的TrackBar位置参数
int g_nthreshold = 100;

//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------

static void on_HoughLines(int, void*);//回调函数

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
	//改变console字体颜色
	system("color 4F");

	//载入原始图和Mat变量定义   
	Mat g_srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图

	//显示原始图  
	imshow("【原始图】", g_srcImage);

	//创建滚动条
	namedWindow("【效果图】", 1);
	createTrackbar("值", "【效果图】", &g_nthreshold, 200, on_HoughLines);

	//进行边缘检测和转化为灰度图
	Canny(g_srcImage, g_midImage, 50, 200, 3);//进行一次canny边缘检测
	cvtColor(g_midImage, g_dstImage, COLOR_GRAY2BGR);//转化边缘检测后的图为灰度图

	//调用一次回调函数，调用一次HoughLinesP函数
	on_HoughLines(g_nthreshold, 0);
	HoughLinesP(g_midImage, g_lines, 1, CV_PI / 180, 80, 50, 10);

	//显示效果图  
	imshow("【效果图】", g_dstImage);


	waitKey(0);

	return 0;

}


//-----------------------------------【on_HoughLines( )函数】--------------------------------
//		描述：【顶帽运算/黑帽运算】窗口的回调函数
//----------------------------------------------------------------------------------------------
static void on_HoughLines(int, void*)
{
	//定义局部变量储存全局变量
	Mat dstImage = g_dstImage.clone();
	Mat midImage = g_midImage.clone();

	//调用HoughLinesP函数
	vector<Vec4i> mylines;
	HoughLinesP(midImage, mylines, 1, CV_PI / 180, g_nthreshold + 1, 50, 10);

	//循环遍历绘制每一条线段
	for (size_t i = 0; i < mylines.size(); i++)
	{
		Vec4i l = mylines[i];
		line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(23, 180, 55), 1, LINE_AA);
	}
	//显示图像
	imshow("【效果图】", dstImage);
}
```

实验结果：
![avatar](\picture\31.霍夫变换.png)

### 7.3 重映射

#### 概念
重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。为了完成映射过程，需要获得一些插值为非整数像素的坐标，因为源图像与目标图像的像素坐标不是一一对应的。一般情况下，我们通过重映射来表达每个像素的位置（x,y)。

#### 实现重映射：remap()函数
remap()函数会根据指定的映射形式，将源图像进行重映射几何变换，基于的公式如下：dst(x,y)=src(map,(x.y),map,(x,y))
需要注意，此函数不支持就地（in-place）操作。看看其原型和参数。
```C++
void remap(InputArray src,outputArraydst,InputArray mapl,InputArray map2,int interpolation,intborderMode=BORDER_CONSTANT,const scalar& bordervalue=scalar ())
```

### 程序：重映射
代码：
```C++
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME "【程序窗口】"        //为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//          描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage;
Mat g_map_x, g_map_y;


//-----------------------------------【全局函数声明部分】--------------------------------------
//          描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
int update_map(int key);

//-----------------------------------【main( )函数】--------------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//改变console字体颜色
	system("color 5F");

	//【1】载入原始图
	g_srcImage = imread("1.jpg", 1);
	if (!g_srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }
	imshow("原始图", g_srcImage);

	//【2】创建和原始图一样的效果图，x重映射图，y重映射图
	g_dstImage.create(g_srcImage.size(), g_srcImage.type());
	g_map_x.create(g_srcImage.size(), CV_32FC1);
	g_map_y.create(g_srcImage.size(), CV_32FC1);

	//【3】创建窗口并显示
	namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME, g_srcImage);

	//【4】轮询按键，更新map_x和map_y的值，进行重映射操作并显示效果图
	while (1)
	{
		//获取键盘按键  
		int key = waitKey(0);

		//判断ESC是否按下，若按下便退出  
		if ((key & 255) == 27)
		{
			cout << "程序退出...........\n";
			break;
		}

		//根据按下的键盘按键来更新 map_x & map_y的值. 然后调用remap( )进行重映射
		update_map(key);
		//此句代码的OpenCV2版为：
		//remap( g_srcImage, g_dstImage, g_map_x, g_map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
		//此句代码的OpenCV3版为：
		remap(g_srcImage, g_dstImage, g_map_x, g_map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

		//显示效果图
		imshow(WINDOW_NAME, g_dstImage);
	}
	return 0;
}

//-----------------------------------【update_map( )函数】--------------------------------
//          描述：根据按键来更新map_x与map_x的值
//----------------------------------------------------------------------------------------------
int update_map(int key)
{
	//双层循环，遍历每一个像素点
	for (int j = 0; j < g_srcImage.rows; j++)
	{
		for (int i = 0; i < g_srcImage.cols; i++)
		{
			switch (key)
			{
			case '1': // 键盘【1】键按下，进行第一种重映射操作
				if (i > g_srcImage.cols * 0.25 && i < g_srcImage.cols * 0.75 && j > g_srcImage.rows * 0.25 && j < g_srcImage.rows * 0.75)
				{
					g_map_x.at<float>(j, i) = static_cast<float>(2 * (i - g_srcImage.cols * 0.25) + 0.5);
					g_map_y.at<float>(j, i) = static_cast<float>(2 * (j - g_srcImage.rows * 0.25) + 0.5);
				}
				else
				{
					g_map_x.at<float>(j, i) = 0;
					g_map_y.at<float>(j, i) = 0;
				}
				break;
			case '2':// 键盘【2】键按下，进行第二种重映射操作
				g_map_x.at<float>(j, i) = static_cast<float>(i);
				g_map_y.at<float>(j, i) = static_cast<float>(g_srcImage.rows - j);
				break;
			case '3':// 键盘【3】键按下，进行第三种重映射操作
				g_map_x.at<float>(j, i) = static_cast<float>(g_srcImage.cols - i);
				g_map_y.at<float>(j, i) = static_cast<float>(j);
				break;
			case '4':// 键盘【4】键按下，进行第四种重映射操作
				g_map_x.at<float>(j, i) = static_cast<float>(g_srcImage.cols - i);
				g_map_y.at<float>(j, i) = static_cast<float>(g_srcImage.rows - j);
				break;
			}
		}
	}
	return 1;
}
```

实验结果：
![avatar](\picture\32.重映射.png)
![avatar](\picture\32.重映射1.png)
![avatar](\picture\32.重映射2.png)
![avatar](\picture\32.重映射3.png)

### 7.4 仿射变换

#### 简介
仿射变换(Affine Transformation或 Affine Map)，又称仿射映射，是指在几何中，一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间的过程。它保持了二维图形的“平直性”(直线经过变换之后依然是直线）和“平行性”(二维图形之间的相对位置关系保持不变，平行线依然是平行线，且直线上点的位置顺序不变)。

一个任意的仿射变换都能表示为乘以一个矩阵（线性变换）接着再加上一个向量（平移）的形式。

那么，我们能够用仿射变换来表示如下三种常见的变换形式:
- 旋转，rotation(线性变换)
- 平移，translation(向量加)
- 缩放，scale(线性变换)

#### 求法
仿射变换表示的就是两幅图片之间的一种联系，关于这种联系的信息大致可从以下两种场景获得。
- 已知X和T，而且已知它们是有联系的。接下来的工作就是求出矩阵M。
- 已知M和X，想求得T。只要应用算式T-=M·X即可。对于这种联系的信息可以用矩阵M清晰地表达(即给出明确的2×3矩阵),也可以用两幅图片点之间几何关系来表达。

####　进行仿射变换: warpAffine()函数

warpAffine函数的作用是依据以下公式子，对图像做仿射变换。
dst (x, y) =src(M 1x+ Mzy+ M3,Mz1x+ Mzzy+ Mzs)
函数原型如下。
```C++
void warpAffine(InputArray src,outputArray dst,InputArray M,Sizedsize,int flags=INTER_LINEAR,intborderMode=BORDER_CONSTANT,constscalar& bordervalue=Scalar ())
```

#### 计算二维旋转变换矩阵: getRotationMatrix2D()函数
getRotationMatrix2D()函数用于计算二维旋转变换矩阵。变换会将旋转中心映射到它自身。
```C++
Mat getRotationMatrix2D(Point2fcenter,double angle,double scale)
```

### 程序：仿射变换
代码：
```C++
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】-------------------------------------------- 
//		描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "【原始图窗口】"					//为窗口标题定义的宏 
#define WINDOW_NAME2 "【经过Warp后的图像】"        //为窗口标题定义的宏 
#define WINDOW_NAME3 "【经过Warp和Rotate后的图像】"        //为窗口标题定义的宏 

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	//【0】改变console字体颜色
	system("color 1F");

	//【1】参数准备
	//定义两组点，代表两个三角形
	Point2f srcTriangle[3];
	Point2f dstTriangle[3];
	//定义一些Mat变量
	Mat rotMat(2, 3, CV_32FC1);
	Mat warpMat(2, 3, CV_32FC1);
	Mat srcImage, dstImage_warp, dstImage_warp_rotate;

	//【2】加载源图像并作一些初始化
	srcImage = imread("1.jpg", 1);
	if (!srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }
	// 设置目标图像的大小和类型与源图像一致
	dstImage_warp = Mat::zeros(srcImage.rows, srcImage.cols, srcImage.type());

	//【3】设置源图像和目标图像上的三组点以计算仿射变换
	srcTriangle[0] = Point2f(0, 0);
	srcTriangle[1] = Point2f(static_cast<float>(srcImage.cols - 1), 0);
	srcTriangle[2] = Point2f(0, static_cast<float>(srcImage.rows - 1));

	dstTriangle[0] = Point2f(static_cast<float>(srcImage.cols * 0.0), static_cast<float>(srcImage.rows * 0.33));
	dstTriangle[1] = Point2f(static_cast<float>(srcImage.cols * 0.65), static_cast<float>(srcImage.rows * 0.35));
	dstTriangle[2] = Point2f(static_cast<float>(srcImage.cols * 0.15), static_cast<float>(srcImage.rows * 0.6));

	//【4】求得仿射变换
	warpMat = getAffineTransform(srcTriangle, dstTriangle);

	//【5】对源图像应用刚刚求得的仿射变换
	warpAffine(srcImage, dstImage_warp, warpMat, dstImage_warp.size());

	//【6】对图像进行缩放后再旋转
	// 计算绕图像中点顺时针旋转50度缩放因子为0.6的旋转矩阵
	Point center = Point(dstImage_warp.cols / 2, dstImage_warp.rows / 2);
	double angle = -50.0;
	double scale = 0.6;
	// 通过上面的旋转细节信息求得旋转矩阵
	rotMat = getRotationMatrix2D(center, angle, scale);
	// 旋转已缩放后的图像
	warpAffine(dstImage_warp, dstImage_warp_rotate, rotMat, dstImage_warp.size());


	//【7】显示结果
	imshow(WINDOW_NAME1, srcImage);
	imshow(WINDOW_NAME2, dstImage_warp);
	imshow(WINDOW_NAME3, dstImage_warp_rotate);

	// 等待用户按任意按键退出程序
	waitKey(0);

	return 0;
}
```

实验结果：
![avatar](\picture\33.仿射变换.png)

### 7.5 直方图均衡化

#### 概念
简而言之，直方图均衡化是通过拉伸像素强度分布范围来增强图像对比度的一种方法。

均衡化处理后的图像只能是近似均匀分布。均衡化图像的动态范围扩大了，但其本质是扩大了量化间隔，而量化级别反而减少了，因此，原来灰度不同的象素经处理后可能变的相同，形成了一片相同灰度的区域，各区域之间有明显的边界，从而出现了伪轮廓。

在原始图像对比度本来就很高的情况下，如果再均衡化则灰度调和，对比度会降低。在泛白缓和的图像中，均衡化会合并一些象素灰度，从而增大对比度。均衡化后的图片如果再对其均衡化，则图像不会有任何变化

#### 实现直方图均衡化: equalizeHist()函数
在OpenCV中，直方图均衡化的功能实现由equalizeHist函数完成。我们一起看看它的函数描述。
```C++
void equalizeHist ( InputArray src, outputArray dst)
```

### 程序：直方图均衡化
代码：
```C++
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	// 【1】加载源图像
	Mat srcImage, dstImage;
	srcImage = imread("1.jpg", 1);
	if (!srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~！ \n"); return false; }

	// 【2】转为灰度图并显示出来
	cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
	imshow("原始图", srcImage);

	// 【3】进行直方图均衡化
	equalizeHist(srcImage, dstImage);

	// 【4】显示结果
	imshow("经过直方图均衡化后的图", dstImage);

	// 等待用户按键退出程序
	waitKey(0);
	return 0;

}
```

实验结果：
![avatar](\picture\34.直方图均衡化.png)