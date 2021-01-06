# <center>AI课程设计报告</center>
## <center>设计题目：基于OpenVino的预训练模型ssd实现车辆识别</center>
## <center> 设计人：胡厚淮</center>
### <center>一、设计背景</center>
无论你是否准备好，智能车的时代都会到来，无人驾驶相比于当下人工驾驶的明显优势让无人驾驶技术成为21世纪的热点技术啊，各个国家和组织都想在这个智能化的浪潮中取得红利，但目前理想的智能无人车时代尚属于起步阶段，虽然很多公司都提出了自己的发展策略，也发布了部分无人车，但距离量产普及还是有一定的差距。在这大背景下，本人结合所学知识，借助intel公司开源深度学习开发工具————Openvino中预训练的模型进行设计了车辆识别功能设计。
### <center>二、设计原理</center>
#### 2.1、设计环境
<br>设计系统：Ubuntu18.04</br>
<br>设计工具：vim文本编辑器、openvino开发工具包、python3.9、Cmake3.15.1</br>
<br>openvino工具介绍：OpenVINO 是英特尔基于自身现有的硬件平台开发的一种可以加快高性能计算机视觉和深度学习视觉应用开发速度工具套件，支持各种英特尔平台的硬件加速器上进行深度学习，并且允许直接异构执行。</br>
<br>openvino的工具组件如下图：</br>

![](./AI%20picture/openvino结构.png)
<br>openvno工具包组件由以下6个部分构成，其中与本次设计相关的有：</br>
- 基于计算机视觉的工具Opencv
- 深度学习推理引擎
- 演示和案例。
    
#### 2.2、设计过程
在Ubuntu上openvino工具包会默认安装在：/opt/intel/openvino_2021.2.185.
-模型下载
<br>Openvino工具提供了专门的模型下载工具 downloader.py，工具的路径在：/opt/intel/openvino_2021.2.185/deployment_tools/tools/model_downloader/</br>
下载模型需要输入对应的参数可以使用：<br>
#python3 model_downloader.py -h 查询。这里有一点需要说明的是，在模型下载的时候，程序会告知下载文件存储路径，但路径不是像Windows那样默认的，有些他会保存在model_downloader路径下，有些是保存在intel_model_zoo下。
-模型训练
模型训练最大的难题是寻找训练集，传统的模型都需要训练，训练时间漫长且电脑容易卡住重来。
-预训练模型转换
openvino开发包提供了专门的模型转换工具model_optimizer，在各个例程里函数使用的模型格式为IR格式。在这种格式下有两个相关文件分别是：</br>
后缀名为:.xml 这类文件会记录模型的拓扑结构、学习率等信息。</br>
后缀名为:.bin 这类文件会记录模型的权重和偏置等信息 </br>
模型转化（模型优化器）文件在：/opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/</br>
转化器能转换大部分的模型，但也有一些模型在转化过程中会出现错误提示。建议详细阅读README.md文件。
-训练模型使用
为了证明设计或者模型训练的有效性，我们需要编辑程序来使用训练的得到的模型，由于对openvino内置函数不了解，这里我使用案例给的程序方案，并在程序中引用了大部分案例代码。</br>
函数设计主要分为4个部分：</br>
- 模型网络的载入</br>
这个模块是，需要从代码输入的参数中读出模型网络，并得网络的特征。

```python
    model = args.model
    log.info(f"Loading network:\n\t{model}")
    net = ie.read_network(model=model)
    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()
```
- 加载资源库</br>
这个模块得到推理引擎，和执行设备信息。

```py
log.info("Device info:")
    versions = ie.get_versions(args.device)
    print("{}{}".format(" " * 8, args.device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[args.device].major,
                                                          versions[args.device].minor))
    print("{}Build ........... {}".format(" " * 8, versions[args.device].build_number))

    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
        log.info("CPU extension loaded: {}".format(args.cpu_extension))
```
- 读入数据</br>
主要是使用opencv模块对图片，视频进行读入和必要的处理。
```py
 for input_key in net.input_info:
        print("input shape: " + str(net.input_info[input_key].input_data.shape))
        print("input key: " + input_key)
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape

    images = np.ndarray(shape=(n, c, h, w))
    images_hw = []
    for i in range(n):
        image = cv2.imread(args.input[i])
        ih, iw = image.shape[:-1]
        images_hw.append((ih, iw))
        log.info("File was added: ")
        log.info("        {}".format(args.input[i]))
        if (ih, iw) != (h, w):
            log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image

```
- 设计分析输出</br>
根据推理结果输出数据。
```py
log.info("Processing output blobs")
    res = res[out_blob]
    boxes, classes = {}, {}
    data = res[0][0]
    for number, proposal in enumerate(data):
        if proposal[2] > 0:
            imid = np.int(proposal[0])
            ih, iw = images_hw[imid]
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            print("[{},{}] element, prob = {:.6}    ({},{})-({},{}) batch id : {}" \
                  .format(number, label, confidence, xmin, ymin, xmax, ymax, imid), end="")
            if proposal[2] > 0.5:
                print(" WILL BE PRINTED!")
                if not imid in boxes.keys():
                    boxes[imid] = []
                boxes[imid].append([xmin, ymin, xmax, ymax])
                if not imid in classes.keys():
                    classes[imid] = []
                classes[imid].append(label)
            else:
                print()

    for imid in classes:
        tmp_image = cv2.imread(args.input[imid])
        for box in boxes[imid]:
            cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
        cv2.imwrite("out.bmp", tmp_image)
        log.info("Image out.bmp created!")
    # -----------------------------------------------------------------------------------------------------

    log.info("Execution successful\n")
```
#### 模型使用讲解
-模型查找</br>
按照官方视频提供的方式我们在使用函数时需要将IR文件所在的目录加载到函数中。我的模型放到了以下位置：
![](./AI%20picture/8.jpg)
-函数执行</br>
我们按照README.md文件要求输入信息执行
![](./AI%20picture/4.jpg)
执行完成后会的到一个图片文件我是在程序里面用系统默认的图片查看器eog查看的。
![](./AI%20picture/6.jpg)
![](./AI%20picture/7.jpg)
#### <center>三、成果展示</center>
我们直接看图下面是两个运行结果
![](./AI%20picture/44.jpeg)
![](./AI%20picture/2.jpg)
![](./AI%20picture/qqq.jpg)
![](./AI%20picture/sendpix0.jpg)
</br>
可以看到两张照片都能被识别，但识别的精度不高，后期还需要加工一下。

#### <center>四、总结</center>
- 本次设计报告针对当前热点无人驾驶背景完成了对车辆识别模块的设计，充分利用了现阶段的有效资源。完成了对ssd模型的有效训练，使用python语言衔接模型和代码可视化。
- 通过这次课程设计，我从中学习到了深度学习开发的基本过程，训练自己实际动手操作能力。
- 巩固了机器视觉开发工具Opencv的函数库，学习了Openvino的使用和开发。

#### <center>五、个人体会</center>
好家伙，终于到了吐槽的环节。先来说一下Openvino的安装吧，当初为了安装成功，我安装的是2019R1.1版本的openvino，也安装成功了，当时还蛮开心的毕竟Windows的环境配置已经失败很多次了，但是在执行测试样例的时候存在SqueezeNet模型下载失败的错误，而且一直是错误的(其实当时我根本不知道那个demo是什么作用)，但是当时饭点到了，吃完回来，心情好了些就直接执行下一步了，也是一路顺风。</br>
很久以后，我要做这个设计报告，openvino能加分，那就搞呗。开始了，刚开始我像一只无头苍蝇一样，在网上搜Openvino的开发教程，可惜真正能带我入门的教程很少(确切的说，我以为openvino会像opencv那样只要学习一些内置函数就行，所以他说的我云里雾里的)。后来我在B站上找到了intel官方视频且有字幕(B站大学果然是存在的)，于是我把那个五十多个视频集看完了。然后把部分视频的操作操作了一遍，于是入门了。</br>
雄赳赳气昂昂，我又来了。那我就下载一个模型跑跑看，真正的秃头操作开始了，我从一些经典的模型开始下载，最后的结果都是FAILED。于是我就在网上GitHub上git了一个squeezenet，但是格式和intel的文件有区别，有一个文件的文件名也不一样，于是使用model_optimizer模块是就疯狂报错，执行多次之后我就放弃了。原以为是版本的问题，所以我把openvino更新到了最新版，但是情况没有变化，但是有一个squeezenet的tensorflow模型是可以下载的，我是用Mo_caffe试图转化为caff模型再转化IR格式，但是他还是报错，前车之鉴果断换一个。</br>
然后就下载了SSD模型，这简直是人间天使(其实intel内部的模型都能下载)，然后我打开了与SSD对应的样例，研究了一下代码。于是就一步步的做出来了。</br>
真正的心得体会再这里，首先我要感谢龚老师设计要求里面说Openvino能加分的要求，使我被动的学习了openvino工具，它真的是一个强大的工具，我还在一篇篇的README.md文件中看到了高端的开发工程师是有很好的解说素质的，尤其是视频介绍的那位工程师，真的是通俗的解释，虽然每个视频都很短但是信息量极大。然后要感谢intel公司开源了这个工具，让小白有一个对深度学习有了一个实战的平台。最后就是对自己的要求了，整个设计过程虽然很锻炼素质，但是也暴露了很多问题，比如说快速学习的能力，openvino的内容很多但是学习如何使用是不难的；还有就是要对学习保持热情，要不然谁熬夜啊；最后就是要主动迎接挑战。
