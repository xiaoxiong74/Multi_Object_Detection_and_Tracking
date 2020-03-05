# A Demo of Multi_Object_Detection_and_Tracking(Counting)
<img src="https://github.com/xiaoxiong74/Multi_Object_Detection_and_Tracking/blob/master/deep_sort_yolov3/output/result.png" width="80%" height="80%"> 
* 部分结果展示
<img src="https://github.com/xiaoxiong74/Multi_Object_Detection_and_Tracking/blob/master/deep_sort_yolov3/output/result.gif" width="80%" height="80%"> 

## Introduction
* __Detection__: __keras-yolo3__ 进行多目标检测训练得到目标检测器
* __Tracking__: __deep_sort_yolov3__ 对检测到的目标进行跟踪

## Requirement
* OpenCV
* keras
* NumPy
* sklean
* Pillow
* tensorflow-gpu 1.10.0

参考:
* 检测器: [keras-yolo3](https://github.com/qqwweee/keras-yolo3)
* 跟踪器:  [deep_sort_yolov3](https://github.com/nwojke/deep_sort)

## Start
__0.安装包__

    pip install -r requirements.txt
    
__1. 克隆项目.__
    
    git clone https://github.com/xiaoxiong74/Multi_Object_Detection_and_Tracking.git

__2. 训练自己的目标检测器__
* 基于keras-yolo3，可以参照此[博客](https://blog.csdn.net/Patrick_Lxc/article/details/80615433)进行基于keras-yolov3进行从0开始
训练自己的目标检测器

__3. 进行目标检测或计数:__
* 修改 `deep_sort_yolov3/model_data/our_classes.txt` 的目标类别为自己训练的目标类别
* 可以修改 `deep_sort_yolov3/yolo.py` 中 `__init__`  初始化的一些参数，如iou阈值、置信度阈值、模型路径等，也可以
在`detect_image` 的for循环中对某些类别做一些返回限制
* 根据自己训练的检测类别数量修改 `deep_sort_yolov3/main.py` 中跟类别有关的参数，可以根据代码中的注释进行对应修改
* 修改完成后直接运行 `python main.py` 即可开始检测

