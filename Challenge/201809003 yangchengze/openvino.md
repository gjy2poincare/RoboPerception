# 基于openvino的人类姿态识别
## 安装openvino
1.去官网上下载下载openvino
2.安装openvino
![yang](/photo/1.jpg)
3.设置好安装路径，即可安装完成。
### 安装依赖项
#### 安装cmake
1.在官网上下载cmake
![yang](/photo/3.png)
2.下载完成之后，安装下载好的cmake
![yang](/photo/2.png)
3.安装完成后即可
#### 安装Python
因为在学校的电脑上进行试验，已经安装好了Python环境。
![yang](/photo/4.jpg)
### 设置环境变量
进入bin文件夹
![yang](/photo/5.png)
运行setupvars.bat
![yang](/photo/6.png)
###  配置模型优化器
进入install_prerequisites文件夹下，运行install_prerequisites.bat。
### 运行示例，识别小车的颜色和车牌号
将找到的车辆图片放进去
![yang](/photo/7.png)
之后运行demo_squeezenet_download_convert_run.bat文件，即可识别出图片中车辆的车牌号和车辆颜色
![yang](/photo/8.png)
由此，openvino可以正常运行
## 进行行人姿态检测试验
### 1.在openvino官网中选择模型
![yang](/photo/9.png)
下载模型到桌面创好的文件夹里
![yang](/photo/10.png)
### 2.模型下载好之后，进入inference_engine_demos_build文件夹中，用vs studio打开Demos.sln文件
运行结果
![yang](/photo/12.png)
### 3.根据网站上的代码在Windows上输入命令
![yang](/photo/13.png)
先进入Debug文件夹中
![yang](/photo/14.png)
输入命令，运行模型
```
human_pose_estimation_demo.exe -i C:\Users\HUAT_IAE\Desktop\pao.mp4 -m C:\Users\HUAT_IAE\Desktop\model\intel\human-pose-estimation-0001\FP32\human-pose-estimation-0001.xml
```
其中C:\Users\HUAT_IAE\Desktop\pao.mp4 是素材的路径，C:\Users\HUAT_IAE\Desktop\model\intel\human-pose-estimation-0001\FP32\human-pose-estimation-0001.xml是模型的路径。
![yang](/photo/15.png)
### 4.得出识别的结果
![yang](/photo/16.gif)
### 代码如下
```
// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>
#include <chrono>

#include <inference_engine.hpp>

#include <monitors/presenter.h>
#include <samples/images_capture.h>
#include <samples/ocv_common.hpp>

#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace InferenceEngine;
using namespace human_pose_estimation;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "InferenceEngine: " << *GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        cv::Mat curr_frame = cap->read();
        if (!curr_frame.data) {
            throw std::runtime_error("Can't read an image from the input");
        }

        cv::Size graphSize{curr_frame.cols / 4, 60}; 
        Presenter presenter(FLAGS_u, curr_frame.rows - graphSize.height - 10, graphSize);

        estimator.re

```
### 总结
openvino工作步骤
1.载入推论引擎插件（Plugin）
2.读取由模型优化产出的中间档
3.配置输入和输出内容
4.载入模型给插件
5.产生推论需求
6.准备输入
7.进行推论
8.处理输出
本文详细叙述了要实现基于openvino检测人类姿态的详细步骤，从准备工作开始，介绍了如何安装openvino及安装openvino的准备工作，如Python等等。最后实现的示例程序的应用，使用示例程序识别出车的类型及车牌号。之后开始进入主题，介绍了下载应用姿态识别模型的详细流程，这其中应用到了visual studio，通过一系列的操作实现了给定一份视频识别出视频中的人类的姿态。
### 心得
本次实验做了用openvino实现人类姿态检测，为了达到这个目标，我做了许多的工作。首先学会了openvino的安装及使用，在安装过程中出现了许多的问题，经常因为依赖包的问题导致openvino无法使用，经过了许多次的安装，终于是openvino的示例运行了出来。最后在运行下载好的模型的过程中，因为不知道该怎么运行导致手足无措，后来看了教程，研究了下载模型时给的例子，终于明白了例子中的命令所代表的意义，实现了人类姿态的识别。