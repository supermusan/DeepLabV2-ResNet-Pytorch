# 74.95-DeepLabV2-ResNet

The project is an reimplementation of [DeepLabV2-ResNet](http://liangchiehchen.com/projects/DeepLabv2_resnet.html) in Pytorch for semantic image segmentation on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/). 

***Attention***: 
* This proj is based on [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet). 
* In this proj, we change the loss-calculate method, which **ignores the background labels**.
* Evaluation of a single-scale model on the PASCAL VOC validation dataset leads to <code>74.95%</code> mIoU [VOC12_50000.pth](https://pan.baidu.com/s/1bP52R8) which is almost equal to <code>75.1</code> reimplemented by [DrSleep](https://github.com/DrSleep/tensorflow-deeplab-resnet). 

![mIoU](https://github.com/CarryJzzZ/74.95-DeepLabV2-ResNet-Pytorch/blob/master/snapshots/mIoU.png)

* The running means and variances of ```batch normalization``` layer of ResNet will be updated. I will try to use ```for i in self.bn.parameters(): i.requires_grad = False```for ResNet layer to verify the performance.
* ***Pytorch*** is more flexible to use **multi-gpu** than ***TensorFlow***, just use```torch.nn.DataParallel(model).cuda()```. But for ***BatchNorm synchronization across multipe GPUs*** I will try it after.

## Usage

## Prerequisites:
1. python 3
2. pytorch 0.3.1
3. numpy
4. opencv

## Train
1. Download this proj ```git clone https://github.com/CarryJzzZ/pithy-conky-colors.git```and enter it.
2. Download the ```init.pth``` which contains MS COCO trained weights. [init.pth](https://pan.baidu.com/s/1McWHataEIpVVsTL45E__8A) and put it into ```dataset``` folder.
3. Change ```DATA_DIRECTORY```**line 24** of ```train.py```  to **VOC2012** where you store the pascal voc12 dataset. (trainning dataset is based on [SBD](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0))
4. run ```python train.py --random-mirror --random-scale --gpu 0```

## Evaluation
1. change ```RESTORE_FROM``` of ```evaluate.py``` to your **trained .pth file** or you can download [demo weights](https://pan.baidu.com/s/1q4dCvuM_pcto2CGARrwgzg)
2. run ```python evaluate.py```
3. predictions are stored in ```outputs```
![ex:prediction](https://github.com/CarryJzzZ/74.95-DeepLabV2-ResNet-Pytorch/blob/master/snapshots/mIoU.png)
