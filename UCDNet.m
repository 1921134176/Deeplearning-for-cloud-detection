function [net,log] = UCDNet(imagesize,imageDir,labelDir,imageDirV,labelDirV)
% A full convolution neural network model based on multi-scale feature fusion for cloud area detection of remote sensing image
% Construct UCDNet model using Matlab Deep Learning Toolbox 14.0

% Author: Cheng Xin, Ocean University of China, Email: chengxin@stu.ouc.edu.cn
% Download URL: https://github.com/1921134176/Deeplearning-for-cloud-detection.git

% imagesize: The size of the imageinput,which format is a vector of 1X3
% imageDir: Training data set folder path
% labelDir: Label data set folder path
% imageDirV: Validation data set folder path
% labelDirV: Validation-label data set folder path

%%
% Create a lgraph.
% Create a lgraph variable to include the network layer.
lgraph = layerGraph();
%%
% Add layer branch
% Add network branches to the lgraph. Each branch is a linear layer group.
tempLayers = [
    imageInputLayer(imagesize,"Name","imageinput","Normalization","rescale-zero-one")
    %imageInputLayer([512 512 3],"Name","imageinput","Normalization","rescale-zero-one")
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_1","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],128,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_2","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    convolution2dLayer([3 3],256,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_3","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_7")
    convolution2dLayer([3 3],512,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_4","Stride",[2 2])
    convolution2dLayer([3 3],1024,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_9")
    convolution2dLayer([3 3],1024,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_10")
    transposedConv2dLayer([4 4],512,"Name","transposed-conv_1","Cropping",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_1")
    convolution2dLayer([3 3],512,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_11")
    convolution2dLayer([3 3],512,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_12")
    transposedConv2dLayer([4 4],256,"Name","transposed-conv_2","Cropping",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_2")
    convolution2dLayer([3 3],256,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")
    reluLayer("Name","relu_13")
    convolution2dLayer([3 3],256,"Name","conv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_14")
    transposedConv2dLayer([4 4],128,"Name","transposed-conv_3","Cropping",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_3")
    convolution2dLayer([3 3],128,"Name","conv_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15")
    reluLayer("Name","relu_15")
    convolution2dLayer([3 3],128,"Name","conv_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_16")
    reluLayer("Name","relu_16")
    transposedConv2dLayer([4 4],64,"Name","transposed-conv_4","Cropping",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_4")
    convolution2dLayer([3 3],64,"Name","conv_17","Padding","same")
    batchNormalizationLayer("Name","batchnorm_17")
    reluLayer("Name","relu_17")
    convolution2dLayer([3 3],64,"Name","conv_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_18")
    reluLayer("Name","relu_18")
    convolution2dLayer([1 1],2,"Name","conv_19","Padding","same")
    softmaxLayer("Name","softmax")
    pixelClassificationLayer("Name","pixel-class")];
lgraph = addLayers(lgraph,tempLayers);

% Clean up auxiliary function variable.
clear tempLayers;
%%
% Connection layer branch.
% Connect all branches of the network to create a network lgraph.
lgraph = connectLayers(lgraph,"relu_2","maxpool_1");
lgraph = connectLayers(lgraph,"relu_2","depthcat_4/in2");
lgraph = connectLayers(lgraph,"relu_4","maxpool_2");
lgraph = connectLayers(lgraph,"relu_4","depthcat_3/in2");
lgraph = connectLayers(lgraph,"relu_6","maxpool_3");
lgraph = connectLayers(lgraph,"relu_6","depthcat_2/in2");
lgraph = connectLayers(lgraph,"relu_8","maxpool_4");
lgraph = connectLayers(lgraph,"relu_8","depthcat_1/in2");
lgraph = connectLayers(lgraph,"transposed-conv_1","depthcat_1/in1");
lgraph = connectLayers(lgraph,"transposed-conv_2","depthcat_2/in1");
lgraph = connectLayers(lgraph,"transposed-conv_3","depthcat_3/in1");
lgraph = connectLayers(lgraph,"transposed-conv_4","depthcat_4/in1");
%%
% Draw layer
plot(lgraph);
%%
% Create training data set
imds = imageDatastore(imageDir);
classNames = ["cloud","background"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
trainingData = pixelLabelImageDatastore(imds,pxds);
%%
% Create validation data set
imdsV = imageDatastore(imageDirV);
pxdsV = pixelLabelDatastore(labelDirV,classNames,labelIDs);
ValidationData = pixelLabelImageDatastore(imdsV,pxdsV);
%%
% Checkpoint file path
pwd=cd;
checkpointpath=fullfile(pwd,'checkpoint');
if ~exist(checkpointpath,'dir')
	mkdir(checkpointpath);
end
% 训练参数
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',2, ...
    'Shuffle','every-epoch',...
    'ValidationData',ValidationData,...
    'Plots','training-progress',...
    'CheckpointPath',checkpointpath);
%%
% Model training
[net,log] = trainNetwork(trainingData,lgraph,options);
end

