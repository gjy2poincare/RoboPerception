# 设计报告

## 基于Opencv的车辆检测

1、传统检测方法

  常规的机器学习方法，包括训练和应用两个过程。

  训练：需要构建训练集（包括正负样本），使用HOG、SIFT等特征描述获取特征，使用SVM（支持向量机）、决策树等对上一步获取的特征和对应的标签（标签指：正样本或者负样本）进行训练（训练指：自动生成SVM或者决策树等的参数，使其可以用来分类）。

  应用：提取需要识别的图片的HOG、SIFT等特征，使用训练好的SVM或者决策树对提取的特征进行分类。

2、神经网络

  通过神经网络训练正负样本，可以直接识别。

  考虑到应用场景为辅助驾驶，神经网络有两个缺陷，第一、神经网络需要大量的并行计算，占用大量的空间，在FPGA、ARM等硬件上运行速度很慢；第二、神经网络本身相当于黑盒，中间数据无法获取、调试起来无从入手，增加了不确定性。所以，这里使用OpenCV进行图像处理。

### 对opencv的使用

由于之前的opencv出现了问题，所以又重新配了一个。

在创建好C++运行环境后，配置包含目录和库目录
![avatar](\img\1.png)
在输入中添加附加依赖项
![avatar](\img\2.png)

### 车辆检测

通过收集数据训练形成数据集，然后通过数据集检测图片上的车辆目标。对图像的检测效果取决于，训练数据集的训练量以及训练集所包含的种类有关，如果想很好的完成图像检测，就需要进行大量的训练。

实现车辆检测，首先需要做一个训练集，这一步得找许多的图片，然后把每一张图片都裁成同样大小，并且其中得包含所需要检测的目标。然后还要准备同样大量的背景图，图片大小也得和之前的一样大，并且其中不能包含目标。

吧所准备的图片分别放到两个不同的文件夹，然后在cmd上分别进入两个文件夹创建样本txt，对两个文件夹做相同的操作，在cmd中输入的语句为“ dir /b/s/p/w *.jpg > neg.txt ”。

输入指令分别生成两个vec文件，在主文件夹下创建一个txt文档，命名为“traincascade”，输入指令后将其后缀名改为bat。然后双击产生训练集xml文件，使用该文件即可进行图像检测。

具体操作的时候，最后一直无法生成xml文件，可能是opencv版本的问题，无奈，只好百度一个xml文件作为训练集。

### 实现代码
代码：
```C++
#include<opencv2\opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

String filename = "E:/Cascade-Classifier-master/cascade700_700_26.xml";  //之前cascade.xml放置的位置
CascadeClassifier car_classifier;

int main(int argc, char** argv)
{
    if (!car_classifier.load(filename))
    {
        printf("could not laod car feature data..\n");
        return -1;
    }

    Mat src = imread("E:/work/4.jpg"); //需要检测的图片
    if (src.empty())
    {
        printf("could not laod image...\n");
        return -1;
    }
    imshow("inupt image", src);
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    vector<Rect>cars;
    car_classifier.detectMultiScale(gray, cars, 1.1, 3, 0, Size(50, 50));
    for (size_t t = 0; t < cars.size(); t++)
    {
        rectangle(src, cars[static_cast<int>(t)], Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow("detect cars", src);
    waitKey(0);
    return 0;

}
```
![avatar](\img\3.png)

使用预先训练好的训练集xml文件，选定其路径，任何把需要检测的图片路径添加到 imread() 中，就可完成对目标图片的检测。

在 rectangle() 中可以选择检测目标框的大小，以及实际所产生的目标检测效果。

### 实验结果

使用网上找的GitHub的数据集，按训练程度以及训练内容分不同组数据集。

第一种数据集：
![avatar](\img\5.png)
![avatar](\img\6.png)

另一组数据集：
![avatar](\img\7.png)
![avatar](\img\8.png)

其他图片效果：
![avatar](\img\9.png)

## 基于Openvino的车辆检测（略）

通过之前安装好的openvino来进行实验。

打开demo文件后，直接运行bat文件即可实现相关功能，前提是需要提前配置好环境文件和所检测的图片。

由于openvino的工具集中有多种可实现不同功能的目标检测代码，可直接拿来使用，非常方便，而且功能很多。

其中最简单的车辆检测文件名字为 demo_security_barrier_camera.bat

### 输入

  名称：输入，形状。[1x3x384x672]--格式为[BxCxHxW]的输入图像，其中：
  - B - 批量大小
  - C - 渠道数量
  - H - 图像高度
  - W--图像宽度 预计颜色顺序为BGR。

### 输出

  网输出形状为blob。[1,1,N,7]，其中N是检测到的边界框的数量。每次检测的格式为[image_id,label,conf,x_min,y_min,x_max,y_max]，其中。

  - image_id--该批图像的ID。
  - label--预测类ID
  - conf--预测类的置信度
  - (x_min, y_min)--左上角边框的坐标。
  - (x_max，y_max)--右下角边界框的坐标。

### 代码

```python
:: Copyright (C) 2018-2019 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

@echo off
setlocal enabledelayedexpansion

set TARGET=CPU
set SAMPLE_OPTIONS=
set BUILD_FOLDER=%USERPROFILE%\Documents\Intel\OpenVINO

:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="-d" (
        set TARGET=%2
        echo target = !TARGET!
        shift
    )
    if "%1"=="-sample-options" (
        set SAMPLE_OPTIONS=%2 %3 %4 %5 %6
        echo sample_options = !SAMPLE_OPTIONS!
        shift
    )
    if "%1"=="-help" (
        echo %~n0%~x0 is security barrier camera demo that showcases three models coming with the product
        echo.
        echo Options:
        echo -d name     Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified
        exit /b
    )
    shift
    goto :input_arguments_loop
)

set ROOT_DIR=%~dp0

set target_image_path=%ROOT_DIR%car_1.bmp


set TARGET_PRECISION=FP16


echo target_precision = !TARGET_PRECISION!

if exist "%ROOT_DIR%..\..\bin\setupvars.bat" (
    call "%ROOT_DIR%..\..\bin\setupvars.bat"
) else (
    echo setupvars.bat is not found, INTEL_OPENVINO_DIR can't be set
    goto error
)

echo INTEL_OPENVINO_DIR is set to %INTEL_OPENVINO_DIR%

:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
   echo Error^: Python is not installed. Please install Python 3.5 ^(64-bit^) or higher from https://www.python.org/downloads/
   goto error
)

:: Check if Python version is equal or higher 3.4
for /F "tokens=* USEBACKQ" %%F IN (`python --version 2^>^&1`) DO (
   set version=%%F
)
echo %var%

for /F "tokens=1,2,3 delims=. " %%a in ("%version%") do (
   set Major=%%b
   set Minor=%%c
)

if "%Major%" geq "3" (
   if "%Minor%" geq "5" (
  set python_ver=okay
   )
)
if not "%python_ver%"=="okay" (
   echo Unsupported Python version. Please install Python 3.5 ^(64-bit^) or higher from https://www.python.org/downloads/
   goto error
)

:: install yaml python modules required for downloader.py
pip3 install --user -r "%ROOT_DIR%..\open_model_zoo\tools\downloader\requirements.in"
if ERRORLEVEL 1 GOTO errorHandling


set models_path=%BUILD_FOLDER%\openvino_models\ir
set models_cache=%BUILD_FOLDER%\openvino_models\cache

if not exist "%models_cache%" (
  mkdir "%models_cache%"
)

set downloader_dir=%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader

for /F "tokens=1,2 usebackq" %%a in ("%ROOT_DIR%demo_security_barrier_camera.conf") do (
   echo python "%downloader_dir%\downloader.py" --name "%%b" --output_dir "%models_path%" --cache_dir "%models_cache%"
   python "%downloader_dir%\downloader.py" --name "%%b" --output_dir "%models_path%" --cache_dir "%models_cache%"

   for /F "tokens=* usebackq" %%d in (
      `python "%downloader_dir%\info_dumper.py" --name "%%b" ^|
         python -c "import sys, json; print(json.load(sys.stdin)[0]['subdirectory'])"`
   ) do (
      set model_args=!model_args! %%a "%models_path%\%%d\%target_precision%\%%b.xml"
   )
)

echo.
echo ###############^|^| Generate VS solution for Inference Engine demos using cmake ^|^|###############
echo.
timeout 3

if "%PROCESSOR_ARCHITECTURE%" == "AMD64" (
   set "PLATFORM=x64"
) else (
   set "PLATFORM=Win32"
)

set VSWHERE="false"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
   set VSWHERE="true"
   cd "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" (
      set VSWHERE="true"
      cd "%ProgramFiles%\Microsoft Visual Studio\Installer"
) else (
   echo "vswhere tool is not found"
)

set MSBUILD_BIN=
set VS_PATH=

if !VSWHERE! == "true" (
   for /f "usebackq tokens=*" %%i in (`vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do (
      set VS_PATH=%%i
   )
   if exist "!VS_PATH!\MSBuild\14.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=!VS_PATH!\MSBuild\14.0\Bin\MSBuild.exe"
   )
   if exist "!VS_PATH!\MSBuild\15.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=!VS_PATH!\MSBuild\15.0\Bin\MSBuild.exe"
   )
   if exist "!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe"
   )
)

if "!MSBUILD_BIN!" == "" (
   if exist "C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
      set "MSBUILD_VERSION=14 2015"
   )
   if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe"
      set "MSBUILD_VERSION=15 2017"
   )
   if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"
      set "MSBUILD_VERSION=15 2017"
   )
   if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe" (
      set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"
      set "MSBUILD_VERSION=15 2017"
   )
) else (
   if not "!MSBUILD_BIN:2019=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=16 2019"
   if not "!MSBUILD_BIN:2017=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=15 2017"
   if not "!MSBUILD_BIN:2015=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=14 2015"
)

if "!MSBUILD_BIN!" == "" (
   echo Build tools for Visual Studio 2015 / 2017 / 2019 cannot be found. If you use Visual Studio 2017 / 2019, please download and install build tools from https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017
   GOTO errorHandling
)

set "SOLUTION_DIR64=%BUILD_FOLDER%\inference_engine_demos_build"

echo Creating Visual Studio !MSBUILD_VERSION! %PLATFORM% files in %SOLUTION_DIR64%... && ^
if exist "%SOLUTION_DIR64%\CMakeCache.txt" del "%SOLUTION_DIR64%\CMakeCache.txt"
cd "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\demos" && cmake -E make_directory "%SOLUTION_DIR64%" && cd "%SOLUTION_DIR64%" && cmake -G "Visual Studio !MSBUILD_VERSION!" -A %PLATFORM% "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\demos"
if ERRORLEVEL 1 GOTO errorHandling

timeout 7
echo.
echo ###############^|^| Build Inference Engine demos using MS Visual Studio (MSBuild.exe) ^|^|###############
echo.
timeout 3
echo "!MSBUILD_BIN!" Demos.sln /p:Configuration=Release /t:security_barrier_camera_demo /clp:ErrorsOnly /m
"!MSBUILD_BIN!" Demos.sln /p:Configuration=Release /t:security_barrier_camera_demo /clp:ErrorsOnly /m
if ERRORLEVEL 1 GOTO errorHandling

timeout 7

:runSample
echo.
echo ###############^|^| Run Inference Engine security barrier camera demo ^|^|###############
echo.
timeout 3
cd "%SOLUTION_DIR64%\intel64\Release"
if not exist security_barrier_camera_demo.exe (
   cd "%INTEL_OPENVINO_DIR%\inference_engine\demos\intel64\Release"
   echo "%INTEL_OPENVINO_DIR%\inference_engine\demos\intel64\Release\security_barrier_camera_demo.exe" -i "%target_image_path%" %model_args% -d !TARGET! -d_va !TARGET! -d_lpr !TARGET! !SAMPLE_OPTIONS!
)
else (
   echo "%SOLUTION_DIR64%\intel64\Release\security_barrier_camera_demo.exe" -i "%target_image_path%" %model_args% -d !TARGET! -d_va !TARGET! -d_lpr !TARGET! !SAMPLE_OPTIONS!
)
security_barrier_camera_demo.exe -i "%target_image_path%" %model_args% ^
                                 -d !TARGET! -d_va !TARGET! -d_lpr !TARGET! !SAMPLE_OPTIONS!
if ERRORLEVEL 1 GOTO errorHandling

echo.
echo ###############^|^| Demo completed successfully ^|^|###############
cd "%ROOT_DIR%"

goto :eof

:errorHandling
echo Error
cd "%ROOT_DIR%"
```

### 实现结果

![avatar](\img\10.png)
![avatar](\img\11.png)

## 分析

无论是实现哪一种目标检测，都需要大量的数据来支撑，一个完整且丰富的数据集，可以最大程度上完成对目标的检测。任意一个目标检测的实现，都有大致上来说是三部分：目标图像、数据集和实现代码。其中最重要的是数据集，数据集可以决定目标，而实现代码有很多都是通用的，主要修改数据集的类型，相同的代码就能实现不同的目标检测。

在国内外都有很多官方的数据集网站，用以完成各种研究，其中大多会提供大量完成剪裁的例子图片集合，但是想要得到最终的数据集还需要自己进行训练生成。

## 总结

目标检测，也叫目标提取，是一种基于目标几何和统计特征的图像分割，它将目标的分割和识别合二为一，其准确性和实时性是整个系统的一项重要能力。尤其是在复杂场景中，需要对多个目标进行实时处理时，目标自动提取和识别就显得特别重要。

所谓的目标检测，就是通过学习目标特征来完成对一类有相同特征的目标进行检测，通过特征来选取目标，所学习的特征集越大，就越能更好的识别这一类目标。

在目标检测过程中，目标图片的大小分辨率会很明显的影响检测的效果，若是分辨率低的图片，虽然感觉上图片很清楚，但是就是无法完成目标检测，另一方面若是目标图片中可选目标过多，有可能会出现无法框定的情况，目标的遮挡等问题，也可以通过训练集的情况，来解决这些问题，实现检测不完全但具有目标特征的目标也能够实现。

## 心得

在实现车辆检测时，由于训练集的内容实在是太少了，导致有很多种的车辆都无法识别出来，只能识别最普通的小轿车，像公交车甚至是跑车都不能识别出来，在改变了训练集后就可以识别出更多的车辆，但是在识别精度上不如只识别小轿车的训练集精度高。

在生成训练集的时候，需要正反训练，同样的图片也得有镜像翻转的两张，以增加识别的准确性。另一方面还得准备一份负样本，即图片中完完全全没有目标特征的图片。结合两个样本训练才能更好的达到目标效果，正负样本的量越大，所能检测识别的目标就能更加准确、多样。

大多数的实现代码都是通用的，只要使用不同的数据集就能够完成不同的目标的检测了，但是想要找一个完整的数据集非常不容易，特别是想找一个训练好的就更不容易了。