# 基于深度学习的遥感影像云区检测模型

MSCFF_V2模型是一种全卷积神经网络模型，参考了论文“Li, Z., Shen, H., Cheng, Q., Liu, Y., You, S., He, Z., 2019. Deep learning based cloud detection for medium and high resolution remote sensing images of different sensors. ISPRS Journal of Photogrammetry and Remote Sensing. 150, 197–212”。MSCFF_V2对论文MSCFF模型中的空洞卷积中卷积核的扩展率进行了修改，并基于Matlab Deep Learning Toolbox 14.0对模型进行了重新构建。

![27-0-00000004 (2)](F:\毕设\论文用图\确定使用图片\27-0-00000004 (2).png)

UCD-Net模型是一种全卷积神经网络模型，参考了论文“ Ronneberger, O., P. Fischer and T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015”。UCD-Net模型对论文U-Net模型进行了修改，添加了Batch Normalization层、修改卷积模式为same、将双线性插值上采样层修改为转置卷积层进行上采样。

![26-ucd-net(2)](F:\毕设\论文用图\确定使用图片\26-ucd-net(2).png)

PictureProcess文件夹给出了数据集制作的批处理程序，可以将遥感图像制作成指定大小的输入图像，用于模型的训练。其中train_label函数通过调用Patch_to_num函数，可以批处理文件夹中的所有图像。