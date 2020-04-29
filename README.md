# 基于深度学习的遥感影像云区检测模型

MSCFF_V2模型是一种全卷积神经网络模型，参考了论文“Li, Z., Shen, H., Cheng, Q., Liu, Y., You, S., He, Z., 2019. Deep learning based cloud detection for medium and high resolution remote sensing images of different sensors. ISPRS Journal of Photogrammetry and Remote Sensing. 150, 197–212”。MSCFF_V2对论文MSCFF模型中的空洞卷积中卷积核的扩展率进行了修改，并基于Matlab Deep Learning Toolbox 14.0对模型进行了重新构建。

UCD-Net模型是一种全卷积神经网络模型，参考了论文“ Ronneberger, O., P. Fischer and T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015”。UCD-Net模型对论文U-Net模型进行了修改，添加了Batch Normalization层、修改卷积模式为same、将双线性插值上采样层修改为转置卷积层进行上采样。