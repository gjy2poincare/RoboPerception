# OpenCV编程入门

## 八、图像轮廓与图像分割修复

### 8.1 图像轮廓

#### 寻找轮廓：findContours()函数
findContours() 函数用于在二值图像中寻找轮廓。
```C++
void findContours(InputOutputArray image,OutputArrayofArrays contours,OutputArray hierarchy,int mode,int method,Point offset=Point ( ) )
```
#### 绘制轮廓：drawContours()函数
drawContours()函数用于在图像中绘制外部或内部轮廓。
```C++
void drawContours(InputOutputArray image，InputArrayOfArrays contours, int contourIdx,const scalar& color, int thickness=1, in tlineType=8,InputArray hierarchy=noArray (),int maxLevel=INT_MAX,Point offset=Point () )
```

### 程序：查找并绘制轮廓
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
#define WINDOW_NAME1 "【原始图窗口】"			//为窗口标题定义的宏 
#define WINDOW_NAME2 "【轮廓图】"					//为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage;
Mat g_grayImage;
int g_nThresh = 80;
int g_nThresh_max = 255;
RNG g_rng(12345);
Mat g_cannyMat_output;
vector<vector<Point>> g_vContours;
vector<Vec4i> g_vHierarchy;


//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void*);


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//【0】改变console字体颜色
	system("color 1F");

	// 加载源图像
	g_srcImage = imread("1.jpg", 1);
	if (!g_srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }

	// 转成灰度并模糊化降噪
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	blur(g_grayImage, g_grayImage, Size(3, 3));

	// 创建窗口
	namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME1, g_srcImage);

	//创建滚动条并初始化
	createTrackbar("canny阈值", WINDOW_NAME1, &g_nThresh, g_nThresh_max, on_ThreshChange);
	on_ThreshChange(0, 0);

	waitKey(0);
	return(0);
}

//-----------------------------------【on_ThreshChange( )函数】------------------------------  
//      描述：回调函数
//----------------------------------------------------------------------------------------------  
void on_ThreshChange(int, void*)
{

	// 用Canny算子检测边缘
	Canny(g_grayImage, g_cannyMat_output, g_nThresh, g_nThresh * 2, 3);

	// 寻找轮廓
	findContours(g_cannyMat_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// 绘出轮廓
	Mat drawing = Mat::zeros(g_cannyMat_output.size(), CV_8UC3);
	for (int i = 0; i < g_vContours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//任意值
		drawContours(drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point());
	}

	// 显示效果图
	imshow(WINDOW_NAME2, drawing);
}
```

实验结果：
![avatar](\picture\35.绘制轮廓.png)

### 8.2 寻找物体的凸包

#### 简介
凸包(Convex Hull）是一个计算几何（图形学）中常见的概念。简单来说，给定二维平面上的点集，凸包就是将最外层的点连接起来构成的凸多边型，它是能包含点集中所有点的。理解物体形状或轮廓的一种比较有用的方法便是计算一个物体的凸包，然后计算其凸缺陷(convexity defects)。很多复杂物体的特性能很好地被这种缺陷表现出来。

#### 寻找凸包：convexHull()函数
convexHull()函数用于寻找图像点集中的凸包，其原型声明如下。
```C++
void convexHull(InputArray points, OutputArray hull, bool clockwise=false, bool returnPoints=true )
```

### 程序：寻找和绘制物体的凸包
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
#define WINDOW_NAME1 "【原始图窗口】"					//为窗口标题定义的宏 
#define WINDOW_NAME2 "【效果图窗口】"					//为窗口标题定义的宏 



//-----------------------------------【全局变量声明部分】--------------------------------------
//  描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_grayImage;
int g_nThresh = 50;
int g_maxThresh = 255;
RNG g_rng(12345);
Mat srcImage_copy = g_srcImage.clone();
Mat g_thresholdImage_output;
vector<vector<Point> > g_vContours;
vector<Vec4i> g_vHierarchy;


//-----------------------------------【全局函数声明部分】--------------------------------------
//   描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void*);

//-----------------------------------【main( )函数】------------------------------------------
//   描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	system("color 3F");

	// 加载源图像
	g_srcImage = imread("1.jpg", 1);

	// 将原图转换成灰度图并进行模糊降
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	blur(g_grayImage, g_grayImage, Size(3, 3));

	// 创建原图窗口并显示
	namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME1, g_srcImage);

	//创建滚动条
	createTrackbar(" 阈值:", WINDOW_NAME1, &g_nThresh, g_maxThresh, on_ThreshChange);
	on_ThreshChange(0, 0);//调用一次进行初始化

	waitKey(0);
	return(0);
}

//-----------------------------------【thresh_callback( )函数】----------------------------------  
//      描述：回调函数
//----------------------------------------------------------------------------------------------  
void on_ThreshChange(int, void*)
{
	// 对图像进行二值化，控制阈值
	threshold(g_grayImage, g_thresholdImage_output, g_nThresh, 255, THRESH_BINARY);

	// 寻找轮廓
	findContours(g_thresholdImage_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// 遍历每个轮廓，寻找其凸包
	vector<vector<Point> >hull(g_vContours.size());
	for (unsigned int i = 0; i < g_vContours.size(); i++)
	{
		convexHull(Mat(g_vContours[i]), hull[i], false);
	}

	// 绘出轮廓及其凸包
	Mat drawing = Mat::zeros(g_thresholdImage_output.size(), CV_8UC3);
	for (unsigned int i = 0; i < g_vContours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
		drawContours(drawing, g_vContours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}

	// 显示效果图
	imshow(WINDOW_NAME2, drawing);
}
```

实验结果：
![avatar](\picture\36.凸包.png)

### 8.3 使用多边形将轮廓包围

#### 返回外部矩形边界：boundingRect()函数
此函数计算并返回指定点集最外面（up-right）的矩形边界。
```C++
Rect boundingRect (InputArray points)
```

#### 寻找最小包围矩形：minAreaRect()函数
此函数用于对给定的2D点集，寻找可旋转的最小面积的包围矩形。
```C++
RotatedRect minAreaRect (InputArray points)
```

#### 寻找最小包围圆形: minEnclosingCircle()函数
minEnclosingCircle函数的功能是利用一种迭代算法，对给定的2D点集，去寻找面积最小的可包围它们的圆形。
```C++
void minEnclosingCircle (InputArray points,Point2f& center,float&radius)
```

#### 用椭圆拟合二维点集: fitEllipse()函数
此函数的作用是用椭圆拟合二维点集。
```C++
RotatedRect fitEllipse(InputArray points)
```

#### 逼近多边形曲线：approxPolyDP()函数
approxPolyDP函数的作用是用指定精度逼近多边形曲线。
```C++
void approxPolyDP (InputArray curve, OutputArray approxCurve,double epsilon,bool closed)
```

### 程序：使用多边形包围轮廓
代码：
```C++
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "【原始图窗口】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【效果图窗口】"        //为窗口标题定义的宏 

//-----------------------------------【全局变量声明部分】--------------------------------------
//  描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage;
Mat g_grayImage;
int g_nThresh = 50;//阈值
int g_nMaxThresh = 255;//阈值最大值
RNG g_rng(12345);//随机数生成器

//-----------------------------------【全局函数声明部分】--------------------------------------
//   描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
void on_ContoursChange(int, void*);

//-----------------------------------【main( )函数】--------------------------------------------
//   描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	//【0】改变console字体颜色
	system("color 1F");

	//【1】载入3通道的原图像
	g_srcImage = imread("1.jpg", 1);
	if (!g_srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }

	//【2】得到原图的灰度图像并进行平滑
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	blur(g_grayImage, g_grayImage, Size(3, 3));

	//【3】创建原始图窗口并显示
	namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME1, g_srcImage);

	//【4】设置滚动条并调用一次回调函数
	createTrackbar(" 阈值:", WINDOW_NAME1, &g_nThresh, g_nMaxThresh, on_ContoursChange);
	on_ContoursChange(0, 0);

	waitKey(0);

	return(0);
}

//----------------------------【on_ContoursChange( )函数】---------------------------------
//      描述：回调函数
//-------------------------------------------------------------------------------------------------  
void on_ContoursChange(int, void*)
{
	//定义一些参数
	Mat threshold_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// 使用Threshold检测边缘
	threshold(g_grayImage, threshold_output, g_nThresh, 255, THRESH_BINARY);

	// 找出轮廓
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// 多边形逼近轮廓 + 获取矩形和圆形边界框
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	//一个循环，遍历所有部分，进行本程序最核心的操作
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//用指定精度逼近多边形曲线 
		boundRect[i] = boundingRect(Mat(contours_poly[i]));//计算点集的最外面（up-right）矩形边界
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);//对给定的 2D点集，寻找最小面积的包围圆形 
	}

	// 绘制多边形轮廓 + 包围的矩形框 + 圆形框
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int unsigned i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//随机设置颜色
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());//绘制轮廓
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);//绘制矩形
		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);//绘制圆
	}

	// 显示效果图窗口
	namedWindow(WINDOW_NAME2, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME2, drawing);
}
```

实验结果：
![avatar](\picture\37.多边形包围.png)

### 8.4 图像的矩

#### 简介
矩函数在图像分析中有着广泛的应用，如模式识别、目标分类、目标识别与方位估计、图像编码与重构等。

一个从一幅数字图形中计算出来的矩集，通常描述了该图像形状的全局特征，并提供了大量的关于该图像不同类型的几何特性信息，比如大小、位置、方向及形状等。图像矩的这种特性描述能力被广泛地应用在各种图像处理、计算机视觉和机器人技术领域的目标识别与方位估计中。

一阶矩与形状有关，二阶矩显示曲线围绕直线平均值的扩展程度，三阶矩则是关于平均值的对称性的测量。由二阶矩和三阶矩可以导出一组共7个不变矩。而不变矩是图像的统计特性，满足平移、伸缩、旋转均不变的不变性，在图像识别领域得到了广泛的应用。

#### 矩的计算: moments()函数
moments()函数用于计算多边形和光栅形状的最高达三阶的所有矩。矩用来计算形状的重心、面积，主轴和其他形状特征，如7Hu不变量等。
```C++
Moments moments(InputArray array, bool binaryImage=false )
```

#### 计算轮廓面积：contourArea()函数
contourArea()函数用于计算整个轮廓或部分轮廓的面积。
```C++
double contourArea(InputArray contour, bool oriented=false)
```

#### 计算轮廓长度：arcLength()函数
arcLength()函数用于计算封闭轮廓的周长或曲线的长度。
```C++
double arcLength(InputArray curve, bool closed)
```

### 程序：查找和绘制图像轮廓矩
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
#define WINDOW_NAME1 "【原始图】"					//为窗口标题定义的宏 
#define WINDOW_NAME2 "【图像轮廓】"        //为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_grayImage;
int g_nThresh = 100;
int g_nMaxThresh = 255;
RNG g_rng(12345);
Mat g_cannyMat_output;
vector<vector<Point> > g_vContours;
vector<Vec4i> g_vHierarchy;

//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void*);

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//【0】改变console字体颜色
	system("color 9F");

	// 读入原图像, 返回3通道图像数据
	g_srcImage = imread("1.jpg", 1);

	// 把原图像转化成灰度图像并进行平滑
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	blur(g_grayImage, g_grayImage, Size(3, 3));

	// 创建新窗口
	namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME1, g_srcImage);

	//创建滚动条并进行初始化
	createTrackbar(" 阈值", WINDOW_NAME1, &g_nThresh, g_nMaxThresh, on_ThreshChange);
	on_ThreshChange(0, 0);

	waitKey(0);
	return(0);
}

//-----------------------------------【on_ThreshChange( )函数】-------------------------------
//		描述：回调函数
//-----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void*)
{
	// 使用Canndy检测边缘
	Canny(g_grayImage, g_cannyMat_output, g_nThresh, g_nThresh * 2, 3);

	// 找到轮廓
	findContours(g_cannyMat_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// 计算矩
	vector<Moments> mu(g_vContours.size());
	for (unsigned int i = 0; i < g_vContours.size(); i++)
	{
		mu[i] = moments(g_vContours[i], false);
	}

	//  计算中心矩
	vector<Point2f> mc(g_vContours.size());
	for (unsigned int i = 0; i < g_vContours.size(); i++)
	{
		mc[i] = Point2f(static_cast<float>(mu[i].m10 / mu[i].m00), static_cast<float>(mu[i].m01 / mu[i].m00));
	}

	// 绘制轮廓
	Mat drawing = Mat::zeros(g_cannyMat_output.size(), CV_8UC3);
	for (unsigned int i = 0; i < g_vContours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//随机生成颜色值
		drawContours(drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point());//绘制外层和内层轮廓
		circle(drawing, mc[i], 4, color, -1, 8, 0);;//绘制圆
	}

	// 显示到窗口中
	namedWindow(WINDOW_NAME2, WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME2, drawing);

	// 通过m00计算轮廓面积并且和OpenCV函数比较
	printf("\t 输出内容: 面积和轮廓长度\n");
	for (unsigned int i = 0; i < g_vContours.size(); i++)
	{
		printf(" >通过m00计算出轮廓[%d]的面积: (M_00) = %.2f \n OpenCV函数计算出的面积=%.2f , 长度: %.2f \n\n", i, mu[i].m00, contourArea(g_vContours[i]), arcLength(g_vContours[i], true));
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
		drawContours(drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point());
		circle(drawing, mc[i], 4, color, -1, 8, 0);
	}
}
```

实验结果：
![avatar](\picture\38.轮廓矩.png)

### 8.5 分水岭算法

#### 简介
分水岭算法，是一种基于拓扑理论的数学形态学的分割方法，其基本思想是把图像看作是测地学上的拓扑地貌，图像中每一点像素的灰度值表示该点的海拔高度，每一个局部极小值及其影响区域称为集水盆，而集水盆的边界则形成分水岭。分水岭的概念和形成可以通过模拟浸入过程来说明:在每一个局部极小值表面，刺穿一个小孔，然后把整个模型慢慢浸入水中，随着浸入的加深，每一个局部极小值的影响域慢慢向外扩展，在两个集水盆汇合处构筑大坝，即形成分水岭。

#### 实现分水岭算法：watershed()函数
函数watershed 实现的分水岭算法是基于标记的分割算法中的一种。在把图像传给函数之前，我们需要大致勾画标记出图像中的期望进行分割的区域，它们被标记为正指数。所以，每一个区域都会被标记为像素值1、2、3等，表示成为一个或者多个连接组件。这些标记的值可以使用 findContours() 函数和 drawContours() 函数由二进制的掩码检索出来。不难理解，这些标记就是即将绘制出来的分割区域的“种子”，而没有标记清楚的区域，被置为0。在函数输出中，每一个标记中的像素被设置为“种子”的值，而区域间的值被设置为-1。
```C++
void watershed (InputArray image,InputOutputArray markers)
```

### 程序：分水岭算法
代码：
```C++
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "【程序窗口1】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【分水岭算法效果图】"        //为窗口标题定义的宏

//-----------------------------------【全局函变量声明部分】--------------------------------------
//		描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_maskImage, g_srcImage;
Point prevPt(-1, -1);

//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
static void on_Mouse(int event, int x, int y, int flags, void*);


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//【0】改变console字体颜色
	system("color 6F");

	//【1】载入原图并显示，初始化掩膜和灰度图
	g_srcImage = imread("1.jpg", 1);
	imshow(WINDOW_NAME1, g_srcImage);
	Mat srcImage, grayImage;
	g_srcImage.copyTo(srcImage);
	cvtColor(g_srcImage, g_maskImage, COLOR_BGR2GRAY);
	cvtColor(g_maskImage, grayImage, COLOR_GRAY2BGR);
	g_maskImage = Scalar::all(0);

	//【2】设置鼠标回调函数
	setMouseCallback(WINDOW_NAME1, on_Mouse, 0);

	//【3】轮询按键，进行处理
	while (1)
	{
		//获取键值
		int c = waitKey(0);

		//若按键键值为ESC时，退出
		if ((char)c == 27)
			break;

		//按键键值为2时，恢复源图
		if ((char)c == '2')
		{
			g_maskImage = Scalar::all(0);
			srcImage.copyTo(g_srcImage);
			imshow("image", g_srcImage);
		}

		//若检测到按键值为1或者空格，则进行处理
		if ((char)c == '1' || (char)c == ' ')
		{
			//定义一些参数
			int i, j, compCount = 0;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;

			//寻找轮廓
			findContours(g_maskImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

			//轮廓为空时的处理
			if (contours.empty())
				continue;

			//拷贝掩膜
			Mat maskImage(g_maskImage.size(), CV_32S);
			maskImage = Scalar::all(0);

			//循环绘制出轮廓
			for (int index = 0; index >= 0; index = hierarchy[index][0], compCount++)
				drawContours(maskImage, contours, index, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);

			//compCount为零时的处理
			if (compCount == 0)
				continue;

			//生成随机颜色
			vector<Vec3b> colorTab;
			for (i = 0; i < compCount; i++)
			{
				int b = theRNG().uniform(0, 255);
				int g = theRNG().uniform(0, 255);
				int r = theRNG().uniform(0, 255);

				colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			}

			//计算处理时间并输出到窗口中
			double dTime = (double)getTickCount();
			watershed(srcImage, maskImage);
			dTime = (double)getTickCount() - dTime;
			printf("\t处理时间 = %gms\n", dTime * 1000. / getTickFrequency());

			//双层循环，将分水岭图像遍历存入watershedImage中
			Mat watershedImage(maskImage.size(), CV_8UC3);
			for (i = 0; i < maskImage.rows; i++)
				for (j = 0; j < maskImage.cols; j++)
				{
					int index = maskImage.at<int>(i, j);
					if (index == -1)
						watershedImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
					else if (index <= 0 || index > compCount)
						watershedImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					else
						watershedImage.at<Vec3b>(i, j) = colorTab[index - 1];
				}

			//混合灰度图和分水岭效果图并显示最终的窗口
			watershedImage = watershedImage * 0.5 + grayImage * 0.5;
			imshow(WINDOW_NAME2, watershedImage);
		}
	}

	return 0;
}


//-----------------------------------【onMouse( )函数】---------------------------------------
//		描述：鼠标消息回调函数
//-----------------------------------------------------------------------------------------------
static void on_Mouse(int event, int x, int y, int flags, void*)
{
	//处理鼠标不在窗口中的情况
	if (x < 0 || x >= g_srcImage.cols || y < 0 || y >= g_srcImage.rows)
		return;

	//处理鼠标左键相关消息
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		prevPt = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		prevPt = Point(x, y);

	//鼠标左键按下并移动，绘制出白色线条
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
		line(g_maskImage, prevPt, pt, Scalar::all(255), 5, 8, 0);
		line(g_srcImage, prevPt, pt, Scalar::all(255), 5, 8, 0);
		prevPt = pt;
		imshow(WINDOW_NAME1, g_srcImage);
	}
}
```

实验结果：
![avatar](\picture\39.分水岭算法.png)

### 8.6 图像修补

#### 简介
图像修复技术简单来说，就是利用那些已经被破坏区域的边缘，即边缘的颜色和结构，繁殖和混合到损坏的图像中，以达到图像修补的目的。

#### 实现图像修补: inpaint()函数
在新版 OpenCV中，图像修补技术由inpaint函数实现，它可以用来从扫描的照片中清除灰尘和划痕，或者从静态图像或视频中去除不需要的物体。其原型声明如下。
```C++
void inpaint(InputArray src, InputArray inpaintMask,OutputArray dst, double inpaintRadius, int flags)
```

### 程序：图像修补
代码：
```C++
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//----------------------------------------------------------------------------------------------
#define WINDOW_NAME0 "【原始图参考】"        //为窗口标题定义的宏 
#define WINDOW_NAME1 "【原始图】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【修补后的效果图】"        //为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//          描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat srcImage0, srcImage1, inpaintMask;
Point previousPoint(-1, -1);//原来的点坐标

//-----------------------------------【On_Mouse( )函数】--------------------------------
//          描述：响应鼠标消息的回调函数
//----------------------------------------------------------------------------------------------
static void On_Mouse(int event, int x, int y, int flags, void*)
{
	//鼠标左键弹起消息
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		previousPoint = Point(-1, -1);
	//鼠标左键按下消息
	else if (event == EVENT_LBUTTONDOWN)
		previousPoint = Point(x, y);
	//鼠标按下并移动，进行绘制
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (previousPoint.x < 0)
			previousPoint = pt;
		//绘制白色线条
		line(inpaintMask, previousPoint, pt, Scalar::all(255), 5, 8, 0);
		line(srcImage1, previousPoint, pt, Scalar::all(255), 5, 8, 0);
		previousPoint = pt;
		imshow(WINDOW_NAME1, srcImage1);
	}
}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//改变console字体颜色
	system("color 2F");

	//载入原始图并进行掩膜的初始化
	Mat srcImage = imread("3.jpg", -1);
	if (!srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~！ \n"); return false; }
	srcImage0 = srcImage.clone();
	srcImage1 = srcImage.clone();
	inpaintMask = Mat::zeros(srcImage1.size(), CV_8U);

	//显示原始图参考
	imshow(WINDOW_NAME0, srcImage0);
	//显示原始图
	imshow(WINDOW_NAME1, srcImage1);
	//设置鼠标回调消息
	setMouseCallback(WINDOW_NAME1, On_Mouse, 0);

	//轮询按键，根据不同的按键进行处理
	while (1)
	{
		//获取按键键值
		char c = (char)waitKey();

		//键值为ESC，程序退出
		if (c == 27)
			break;

		//键值为2，恢复成原始图像
		if (c == '2')
		{
			inpaintMask = Scalar::all(0);
			srcImage.copyTo(srcImage1);
			imshow(WINDOW_NAME1, srcImage1);
		}

		//键值为1或者空格，进行图像修补操作
		if (c == '1' || c == ' ')
		{
			Mat inpaintedImage;
			inpaint(srcImage1, inpaintMask, inpaintedImage, 3, INPAINT_TELEA);
			imshow(WINDOW_NAME2, inpaintedImage);
		}
	}

	return 0;
}
```

实验结果：
![avatar](\picture\40.图像修补.png)

## 九、直方图与分配

### 9.1 直方图概述

#### 简介
简单来说，直方图就是对数据进行统计的一种方法，并且将统计值组织到一系列事先定义好的bin当中。其中，bin为直方图中经常用到的一个概念，可翻译为“直条”或“组距”，其数值是从数据中计算出的特征统计量，这些数据可以是诸如梯度、方向、色彩或任何其他特征。且无论如何，直方图获得的是数据分布的统计图。通常直方图的维数要低于原始数据。总而言之，直方图是计算机视觉中最经典的工具之一。

图像直方图(Image Histogram）是用以表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素数。可以借助观察该直方图了解需要如何调整亮度分布。这种直方图中，横坐标的左侧为纯黑、较暗的区域，而右侧为较亮、纯白的区域。因此，一张较暗图片的图像直方图中的数据多集中于左侧和中间部分，而整体明亮、只有少量阴影的图像则相反。计算机视觉领域常借助图像直方图来实现图像的二值化。

直方图的意义如下。
- 直方图是图像中像素强度分布的图形表达方式。
- 它统计了每一个强度值所具有的像素个数。

### 9.2 直方图的计算与绘制

#### 计算直方图: calcHist()函数
在OpenCV中，calcHist()函数用于计算一个或者多个阵列的直方图。原型如下。
```C++
void calcHist (const Mat* images,int nimages,const int* channels,InputArray mask,OutputArray hist,int dims,const int* histSize,constfloat** ranges,bool uniform=true,bool accumulate=false )
```

#### 找寻最值: minMaxLoc()函数
minMaxLoc()函数的作用是在数组中找到全局最小值和最大值。它有两个版本的原型，在此介绍常用的那一个版本。
```C++
void minMaxLoc(InputArray src, double* minval, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray () )
```

### 程序：绘制H-S直方图
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

	//【1】载入源图，转化为HSV颜色模型
	Mat srcImage, hsvImage;
	srcImage = imread("3.jpg");
	cvtColor(srcImage, hsvImage, COLOR_BGR2HSV);

	system("color 2F");

	//【2】参数准备
	//将色调量化为30个等级，将饱和度量化为32个等级
	int hueBinNum = 30;//色调的直方图直条数量
	int saturationBinNum = 32;//饱和度的直方图直条数量
	int histSize[] = { hueBinNum, saturationBinNum };
	// 定义色调的变化范围为0到179
	float hueRanges[] = { 0, 180 };
	//定义饱和度的变化范围为0（黑、白、灰）到255（纯光谱颜色）
	float saturationRanges[] = { 0, 256 };
	const float* ranges[] = { hueRanges, saturationRanges };
	MatND dstHist;
	//参数准备，calcHist函数中将计算第0通道和第1通道的直方图
	int channels[] = { 0, 1 };

	//【3】正式调用calcHist，进行直方图计算
	calcHist(&hsvImage,//输入的数组
		1, //数组个数为1
		channels,//通道索引
		Mat(), //不使用掩膜
		dstHist, //输出的目标直方图
		2, //需要计算的直方图的维度为2
		histSize, //存放每个维度的直方图尺寸的数组
		ranges,//每一维数值的取值范围数组
		true, // 指示直方图是否均匀的标识符，true表示均匀的直方图
		false);//累计标识符，false表示直方图在配置阶段会被清零

	//【4】为绘制直方图准备参数
	double maxValue = 0;//最大值
	minMaxLoc(dstHist, 0, &maxValue, 0, 0);//查找数组和子数组的全局最小值和最大值存入maxValue中
	int scale = 10;
	Mat histImg = Mat::zeros(saturationBinNum * scale, hueBinNum * 10, CV_8UC3);

	//【5】双层循环，进行直方图绘制
	for (int hue = 0; hue < hueBinNum; hue++)
		for (int saturation = 0; saturation < saturationBinNum; saturation++)
		{
			float binValue = dstHist.at<float>(hue, saturation);//直方图组距的值
			int intensity = cvRound(binValue * 255 / maxValue);//强度

			//正式进行绘制
			rectangle(histImg, Point(hue * scale, saturation * scale),
				Point((hue + 1) * scale - 1, (saturation + 1) * scale - 1),
				Scalar::all(intensity), FILLED);
		}

	//【6】显示效果图
	imshow("素材图", srcImage);
	imshow("H-S 直方图", histImg);

	waitKey();
}
```

实验结果：
![avatar](\picture\41.H-S直方图.png)

### 程序：计算并绘制图像一维直方图
代码：
```C++
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-------------------------------------------------------------------------------------------------
int main()
{
	//【1】载入原图并显示
	Mat srcImage = imread("1.jpg", 0);
	imshow("原图", srcImage);
	if (!srcImage.data) { cout << "fail to load image" << endl; 	return 0; }

	system("color 1F");

	//【2】定义变量
	MatND dstHist;       // 在cv中用CvHistogram *hist = cvCreateHist
	int dims = 1;
	float hranges[] = { 0, 255 };
	const float* ranges[] = { hranges };   // 这里需要为const类型
	int size = 256;
	int channels = 0;

	//【3】计算图像的直方图
	calcHist(&srcImage, 1, &channels, Mat(), dstHist, dims, &size, ranges);    // cv 中是cvCalcHist
	int scale = 1;

	Mat dstImage(size * scale, size, CV_8U, Scalar(0));
	//【4】获取最大值和最小值
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);  //  在cv中用的是cvGetMinMaxHistValue

	//【5】绘制出直方图
	int hpt = saturate_cast<int>(0.9 * size);
	for (int i = 0; i < 256; i++)
	{
		float binValue = dstHist.at<float>(i);           //   注意hist中是float类型    而在OpenCV1.0版中用cvQueryHistValue_1D
		int realValue = saturate_cast<int>(binValue * hpt / maxValue);
		rectangle(dstImage, Point(i * scale, size - 1), Point((i + 1) * scale - 1, size - realValue), Scalar(255));
	}
	imshow("一维直方图", dstImage);
	waitKey(0);
	return 0;
}
```

实验结果：
![avatar](\picture\42.一维直方图.png)

### 程序：绘制RGB三色直方图
代码：
```C++
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
using namespace cv;

//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{

	//【1】载入素材图并显示
	Mat srcImage;
	srcImage = imread("1.jpg");
	imshow("素材图", srcImage);

	system("color 3F");

	//【2】参数准备
	int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	MatND redHist, grayHist, blueHist;
	int channels_r[] = { 0 };

	//【3】进行直方图的计算（红色分量部分）
	calcHist(&srcImage, 1, channels_r, Mat(), //不使用掩膜
		redHist, 1, hist_size, ranges,
		true, false);

	//【4】进行直方图的计算（绿色分量部分）
	int channels_g[] = { 1 };
	calcHist(&srcImage, 1, channels_g, Mat(), // do not use mask
		grayHist, 1, hist_size, ranges,
		true, // the histogram is uniform
		false);

	//【5】进行直方图的计算（蓝色分量部分）
	int channels_b[] = { 2 };
	calcHist(&srcImage, 1, channels_b, Mat(), // do not use mask
		blueHist, 1, hist_size, ranges,
		true, // the histogram is uniform
		false);

	//-----------------------绘制出三色直方图------------------------
	//参数准备
	double maxValue_red, maxValue_green, maxValue_blue;
	minMaxLoc(redHist, 0, &maxValue_red, 0, 0);
	minMaxLoc(grayHist, 0, &maxValue_green, 0, 0);
	minMaxLoc(blueHist, 0, &maxValue_blue, 0, 0);
	int scale = 1;
	int histHeight = 256;
	Mat histImage = Mat::zeros(histHeight, bins * 3, CV_8UC3);

	//正式开始绘制
	for (int i = 0; i < bins; i++)
	{
		//参数准备
		float binValue_red = redHist.at<float>(i);
		float binValue_green = grayHist.at<float>(i);
		float binValue_blue = blueHist.at<float>(i);
		int intensity_red = cvRound(binValue_red * histHeight / maxValue_red);  //要绘制的高度
		int intensity_green = cvRound(binValue_green * histHeight / maxValue_green);  //要绘制的高度
		int intensity_blue = cvRound(binValue_blue * histHeight / maxValue_blue);  //要绘制的高度

		//绘制红色分量的直方图
		rectangle(histImage, Point(i * scale, histHeight - 1),
			Point((i + 1) * scale - 1, histHeight - intensity_red),
			Scalar(255, 0, 0));

		//绘制绿色分量的直方图
		rectangle(histImage, Point((i + bins) * scale, histHeight - 1),
			Point((i + bins + 1) * scale - 1, histHeight - intensity_green),
			Scalar(0, 255, 0));

		//绘制蓝色分量的直方图
		rectangle(histImage, Point((i + bins * 2) * scale, histHeight - 1),
			Point((i + bins * 2 + 1) * scale - 1, histHeight - intensity_blue),
			Scalar(0, 0, 255));

	}

	//在窗口中显示出绘制好的直方图
	imshow("图像的RGB直方图", histImage);
	waitKey(0);
	return 0;
}
```

实验结果：
![avatar](\picture\43.三色直方图.png)

### 9.3 直方图对比

#### 对比直方图: compareHist()函数
compareHist()函数用于对两幅直方图进行比较。有两个版本的C++原型，如下。
```C++
double compareHist (InputArray Hl, InputArray H2, int method)
```
```C++
double compareHist (const SparseMat& Hl, const SparseMat& H2, int method)
```

### 程序：直方图对比
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
	//【0】改变console字体颜色
	system("color 2F");

	//【1】声明储存基准图像和另外两张对比图像的矩阵( RGB 和 HSV )
	Mat srcImage_base, hsvImage_base;
	Mat srcImage_test1, hsvImage_test1;
	Mat srcImage_test2, hsvImage_test2;
	Mat hsvImage_halfDown;

	//【2】载入基准图像(srcImage_base) 和两张测试图像srcImage_test1、srcImage_test2，并显示
	srcImage_base = imread("1.jpg", 1);
	srcImage_test1 = imread("2.jpg", 1);
	srcImage_test2 = imread("3.jpg", 1);
	//显示载入的3张图像
	imshow("基准图像", srcImage_base);
	imshow("测试图像1", srcImage_test1);
	imshow("测试图像2", srcImage_test2);

	// 【3】将图像由BGR色彩空间转换到 HSV色彩空间
	cvtColor(srcImage_base, hsvImage_base, COLOR_BGR2HSV);
	cvtColor(srcImage_test1, hsvImage_test1, COLOR_BGR2HSV);
	cvtColor(srcImage_test2, hsvImage_test2, COLOR_BGR2HSV);

	//【4】创建包含基准图像下半部的半身图像(HSV格式)
	hsvImage_halfDown = hsvImage_base(Range(hsvImage_base.rows / 2, hsvImage_base.rows - 1), Range(0, hsvImage_base.cols - 1));

	//【5】初始化计算直方图需要的实参
	// 对hue通道使用30个bin,对saturatoin通道使用32个bin
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	// hue的取值范围从0到256, saturation取值范围从0到180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };
	const float* ranges[] = { h_ranges, s_ranges };
	// 使用第0和第1通道
	int channels[] = { 0, 1 };

	// 【6】创建储存直方图的 MatND 类的实例:
	MatND baseHist;
	MatND halfDownHist;
	MatND testHist1;
	MatND testHist2;

	// 【7】计算基准图像，两张测试图像，半身基准图像的HSV直方图:
	calcHist(&hsvImage_base, 1, channels, Mat(), baseHist, 2, histSize, ranges, true, false);
	normalize(baseHist, baseHist, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsvImage_halfDown, 1, channels, Mat(), halfDownHist, 2, histSize, ranges, true, false);
	normalize(halfDownHist, halfDownHist, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsvImage_test1, 1, channels, Mat(), testHist1, 2, histSize, ranges, true, false);
	normalize(testHist1, testHist1, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsvImage_test2, 1, channels, Mat(), testHist2, 2, histSize, ranges, true, false);
	normalize(testHist2, testHist2, 0, 1, NORM_MINMAX, -1, Mat());


	//【8】按顺序使用4种对比标准将基准图像的直方图与其余各直方图进行对比:
	for (int i = 0; i < 4; i++)
	{
		//进行图像直方图的对比
		int compare_method = i;
		double base_base = compareHist(baseHist, baseHist, compare_method);
		double base_half = compareHist(baseHist, halfDownHist, compare_method);
		double base_test1 = compareHist(baseHist, testHist1, compare_method);
		double base_test2 = compareHist(baseHist, testHist2, compare_method);
		//输出结果
		printf(" 方法 [%d] 的匹配结果如下：\n\n 【基准图 - 基准图】：%f, 【基准图 - 半身图】：%f,【基准图 - 测试图1】： %f, 【基准图 - 测试图2】：%f \n-----------------------------------------------------------------\n", i, base_base, base_half, base_test1, base_test2);
	}

	printf("检测结束。");
	waitKey(0);
	return 0;
}
```

实验结果：
![avatar](\picture\44.直方图对比.png)

### 9.4 反向投影

#### 简介
反向投影（back projection）就是一种记录给定图像中的像素点如何适应直方图模型像素分布方式的一种方法。

简单的讲，所谓反向投影就是首先计算某一特征的直方图模型，然后使用模型去寻找图像中存在的该特征的方法。

#### 作用与结果
反向投影用于在输入图像（通常较大）中查找与特定图像（通常较小或者仅1个像素，以下将其称为模板图像）最匹配的点或者区域，也就是定位模板图像出现在输入图像的位置。

反向投影的结果包含了以每个输入图像像素点为起点的直方图对比结果。可以把它看成是一个二维的浮点型数组、二维矩阵，或者单通道的浮点型图像。

#### 计算反向投影：calcBackProject()函数
calcBackProject()函数用于计算直方图的反向投影。
```C++
void calcBackProject (
    const Mat* images,
    int nimages,
    const int* channels,InputArray hist,
    OutputArray backProject,
    const float* * ranges,
    double scale=1,
    bool uniform=true )
```

#### 通道复制：mixChannels()函数
此函数由输入参数复制某通道到输出参数特定的通道中。有两个版本的C++原型，采用函数注释方式分别介绍如下。
```C++
void mixChannels (
    Mat* src,//输入的数组，所有的矩阵必须有相同的尺寸和深度
    size_t nsrcs,//第一个参数src输入的矩阵数
    Mat* dst,//输出的数组，所有矩阵必须被初始化，且大小和深度必须与src[0]相同
    size_t ndsts,//第三个参数dst输入的矩阵数
    const int* fromTo,//对指定的通道进行复制的效组索引
    size_t npairs)//第五个参数fromTo的索引数
```
```C++
void mixChannels(
    const vector<Mat>& src,//输入的矩阵向量，所有的矩阵必须有相同的尺寸和深度
    vector<Mat>& dst,//输出的矩阵向量,所有矩阵须被初始化，且大小和深度须与src[0]相同
    const int* fromTo,//对指定的通道进行复制的数组索引
    size_t npairs)//第三个参数fromTo的索引数
```

### 程序：反向投影
代码：
```C++
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;


//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "【原始图】"        //为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//          描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_hsvImage; Mat g_hueImage;
int g_bins = 30;//直方图组距

//-----------------------------------【全局函数声明部分】--------------------------------------
//          描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void on_BinChange(int, void*);

//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	//【0】改变console字体颜色
	system("color 6F");

	//【1】读取源图像，并转换到 HSV 空间
	g_srcImage = imread("1.jpg", 1);
	if (!g_srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~！ \n"); return false; }
	cvtColor(g_srcImage, g_hsvImage, COLOR_BGR2HSV);

	//【2】分离 Hue 色调通道
	g_hueImage.create(g_hsvImage.size(), g_hsvImage.depth());
	int ch[] = { 0, 0 };
	mixChannels(&g_hsvImage, 1, &g_hueImage, 1, ch, 1);

	//【3】创建 Trackbar 来输入bin的数目
	namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
	createTrackbar("色调组距 ", WINDOW_NAME1, &g_bins, 180, on_BinChange);
	on_BinChange(0, 0);//进行一次初始化

	//【4】显示效果图
	imshow(WINDOW_NAME1, g_srcImage);

	// 等待用户按键
	waitKey(0);
	return 0;
}


//-----------------------------------【on_HoughLines( )函数】--------------------------------
//          描述：响应滑动条移动消息的回调函数
//---------------------------------------------------------------------------------------------
void on_BinChange(int, void*)
{
	//【1】参数准备
	MatND hist;
	int histSize = MAX(g_bins, 2);
	float hue_range[] = { 0, 180 };
	const float* ranges = { hue_range };

	//【2】计算直方图并归一化
	calcHist(&g_hueImage, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	//【3】计算反向投影
	MatND backproj;
	calcBackProject(&g_hueImage, 1, 0, hist, backproj, &ranges, 1, true);

	//【4】显示反向投影
	imshow("反向投影图", backproj);

	//【5】绘制直方图的参数准备
	int w = 400; int h = 400;
	int bin_w = cvRound((double)w / histSize);
	Mat histImg = Mat::zeros(w, h, CV_8UC3);

	//【6】绘制直方图
	for (int i = 0; i < g_bins; i++)
	{
		rectangle(histImg, Point(i * bin_w, h), Point((i + 1) * bin_w, h - cvRound(hist.at<float>(i) * h / 255.0)), Scalar(100, 123, 255), -1);
	}

	//【7】显示直方图窗口
	imshow("直方图", histImg);
}
```

实验结果：
![avatar](\picture\45.反向投影.png)

### 9.5 模板匹配

#### 概念与原理
模板匹配是一项在一幅图像中寻找与另一幅模板图像最匹配（相似）部分的技术。在 OpenCV2 和 OpenCV3 中，模板匹配由 MatchTemplate() 函数完成。需要注意，模板匹配不是基于直方图的，而是通过在输入图像上滑动图像块，对实际的图像块和输入图像进行匹配的一种匹配方法。

#### 实现模板匹配: matchTemplate()函数
matchTemplate()用于匹配出和模板重叠的图像区域。
```C++
void matchTemplate(InputArray image,InputArray templ,OutputArray result, int method)
```

### 程序：模板匹配
代码：
```C++
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;


//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "【原始图片】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【匹配窗口】"        //为窗口标题定义的宏 

//-----------------------------------【全局变量声明部分】------------------------------------
//          描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_templateImage; Mat g_resultImage;
int g_nMatchMethod;
int g_nMaxTrackbarNum = 5;

//-----------------------------------【全局函数声明部分】--------------------------------------
//          描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
void on_Matching(int, void*);


//-----------------------------------【main( )函数】--------------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	//【0】改变console字体颜色
	system("color 1F");

	//【1】载入原图像和模板块
	g_srcImage = imread("1.jpg", 1);
	g_templateImage = imread("4.jpg", 1);

	//【2】创建窗口
	namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
	namedWindow(WINDOW_NAME2, WINDOW_AUTOSIZE);

	//【3】创建滑动条并进行一次初始化
	createTrackbar("方法", WINDOW_NAME1, &g_nMatchMethod, g_nMaxTrackbarNum, on_Matching);
	on_Matching(0, 0);

	waitKey(0);
	return 0;

}

//-----------------------------------【on_Matching( )函数】--------------------------------
//          描述：回调函数
//-------------------------------------------------------------------------------------------
void on_Matching(int, void*)
{
	//【1】给局部变量初始化
	Mat srcImage;
	g_srcImage.copyTo(srcImage);

	//【2】初始化用于结果输出的矩阵
	int resultImage_rows = g_srcImage.rows - g_templateImage.rows + 1;
	int resultImage_cols = g_srcImage.cols - g_templateImage.cols + 1;
	g_resultImage.create(resultImage_rows, resultImage_cols, CV_32FC1);

	//【3】进行匹配和标准化
	matchTemplate(g_srcImage, g_templateImage, g_resultImage, g_nMatchMethod);
	normalize(g_resultImage, g_resultImage, 0, 1, NORM_MINMAX, -1, Mat());

	//【4】通过函数 minMaxLoc 定位最匹配的位置
	double minValue; double maxValue; Point minLocation; Point maxLocation;
	Point matchLocation;
	minMaxLoc(g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, Mat());

	//【5】对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值有着更高的匹配结果. 而其余的方法, 数值越大匹配效果越好
	//此句代码的OpenCV2版为：
	//if( g_nMatchMethod  == CV_TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED )
	//此句代码的OpenCV3版为：
	if (g_nMatchMethod == TM_SQDIFF || g_nMatchMethod == TM_SQDIFF_NORMED)
	{
		matchLocation = minLocation;
	}
	else
	{
		matchLocation = maxLocation;
	}

	//【6】绘制出矩形，并显示最终结果
	rectangle(srcImage, matchLocation, Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), Scalar(0, 0, 255), 2, 8, 0);
	rectangle(g_resultImage, matchLocation, Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), Scalar(0, 0, 255), 2, 8, 0);

	imshow(WINDOW_NAME1, srcImage);
	imshow(WINDOW_NAME2, g_resultImage);

}
```

实验结果：
![avatar](\picture\46.模板匹配1.png)
![avatar](\picture\46.模板匹配2.png)
![avatar](\picture\46.模板匹配3.png)
![avatar](\picture\46.模板匹配4.png)
![avatar](\picture\46.模板匹配5.png)