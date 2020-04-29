function [net,log] = MSCFF_V2(imagesize,imageDir,labelDir,imageDirV,labelDirV)
% A full convolution neural network model based on multi-scale feature fusion for cloud area detection of remote sensing image
% Construct MSCFF_V2 model using Matlab Deep Learning Toolbox 14.0

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
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_1","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    convolution2dLayer([3 3],128,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_2","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_8")
    convolution2dLayer([3 3],256,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_3","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_11")
    convolution2dLayer([3 3],512,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")
    reluLayer("Name","relu_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_14","DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_14")
    convolution2dLayer([3 3],512,"Name","conv_15","DilationFactor",[5 5],"Padding","same")
    batchNormalizationLayer("Name","batchnorm_15")
    reluLayer("Name","relu_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_16")
    reluLayer("Name","relu_16")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_17","DilationFactor",[3 3],"Padding","same")
    batchNormalizationLayer("Name","batchnorm_17")
    reluLayer("Name","relu_17")
    convolution2dLayer([3 3],512,"Name","conv_18","DilationFactor",[5 5],"Padding","same")
    batchNormalizationLayer("Name","batchnorm_18")
    reluLayer("Name","relu_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_19","Padding","same")
    batchNormalizationLayer("Name","batchnorm_19")
    reluLayer("Name","relu_19")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_20","DilationFactor",[3 3],"Padding","same")
    batchNormalizationLayer("Name","batchnorm_20")
    reluLayer("Name","relu_20")
    convolution2dLayer([3 3],512,"Name","conv_21","DilationFactor",[5 5],"Padding","same")
    batchNormalizationLayer("Name","batchnorm_21")
    reluLayer("Name","relu_21")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],2,"Name","conv_42","Padding","same")
    batchNormalizationLayer("Name","batchnorm_42")
    reluLayer("Name","relu_42")
    transposedConv2dLayer([16 16],2,"Name","transposed-conv_8","Cropping",[4 4 4 4],"Stride",[8 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_22","Padding","same")
    batchNormalizationLayer("Name","batchnorm_22")
    reluLayer("Name","relu_22")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_23","DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","batchnorm_23")
    reluLayer("Name","relu_23")
    convolution2dLayer([3 3],512,"Name","conv_24","DilationFactor",[5 5],"Padding","same")
    batchNormalizationLayer("Name","batchnorm_24")
    reluLayer("Name","relu_24")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_25","Padding","same")
    batchNormalizationLayer("Name","batchnorm_25")
    reluLayer("Name","relu_25")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],2,"Name","conv_41","Padding","same")
    batchNormalizationLayer("Name","batchnorm_41")
    reluLayer("Name","relu_41")
    transposedConv2dLayer([16 16],2,"Name","transposed-conv_7","Cropping",[4 4 4 4],"Stride",[8 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_26","Padding","same")
    batchNormalizationLayer("Name","batchnorm_26")
    reluLayer("Name","relu_26")
    convolution2dLayer([3 3],512,"Name","conv_27","Padding","same")
    batchNormalizationLayer("Name","batchnorm_27")
    reluLayer("Name","relu_27")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    transposedConv2dLayer([4 4],512,"Name","transposed-conv_1","Cropping",[1 1 1 1],"Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv_28","Padding","same")
    batchNormalizationLayer("Name","batchnorm_28")
    reluLayer("Name","relu_28")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],2,"Name","conv_40","Padding","same")
    batchNormalizationLayer("Name","batchnorm_40")
    reluLayer("Name","relu_40")
    transposedConv2dLayer([16 16],2,"Name","transposed-conv_6","Cropping",[4 4 4 4],"Stride",[8 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_29","Padding","same")
    batchNormalizationLayer("Name","batchnorm_29")
    reluLayer("Name","relu_29")
    convolution2dLayer([3 3],256,"Name","conv_30","Padding","same")
    batchNormalizationLayer("Name","batchnorm_30")
    reluLayer("Name","relu_30")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],2,"Name","conv_39","Padding","same")
    batchNormalizationLayer("Name","batchnorm_39")
    reluLayer("Name","relu_39")
    transposedConv2dLayer([8 8],2,"Name","transposed-conv_5","Cropping",[2 2 2 2],"Stride",[4 4])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    transposedConv2dLayer([4 4],512,"Name","transposed-conv_2","Cropping",[1 1 1 1],"Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_31","Padding","same")
    batchNormalizationLayer("Name","batchnorm_31")
    reluLayer("Name","relu_31")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","conv_32","Padding","same")
    batchNormalizationLayer("Name","batchnorm_32")
    reluLayer("Name","relu_32")
    convolution2dLayer([3 3],128,"Name","conv_33","Padding","same")
    batchNormalizationLayer("Name","batchnorm_33")
    reluLayer("Name","relu_33")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    transposedConv2dLayer([4 4],512,"Name","transposed-conv_3","Cropping",[1 1 1 1],"Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv_34","Padding","same")
    batchNormalizationLayer("Name","batchnorm_34")
    reluLayer("Name","relu_34")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_35","Padding","same")
    batchNormalizationLayer("Name","batchnorm_35")
    reluLayer("Name","relu_35")
    convolution2dLayer([3 3],64,"Name","conv_36","Padding","same")
    batchNormalizationLayer("Name","batchnorm_36")
    reluLayer("Name","relu_36")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],2,"Name","conv_38","Padding","same")
    batchNormalizationLayer("Name","batchnorm_38")
    reluLayer("Name","relu_38")
    transposedConv2dLayer([4 4],2,"Name","transposed-conv_4","Cropping",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_17")
    convolution2dLayer([3 3],2,"Name","conv_37","Padding","same")
    batchNormalizationLayer("Name","batchnorm_37")
    reluLayer("Name","relu_37")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(6,"Name","depthcat")
    convolution2dLayer([3 3],2,"Name","conv_43","Padding","same")
    softmaxLayer("Name","softmax")
    pixelClassificationLayer("Name","pixel-class")];
lgraph = addLayers(lgraph,tempLayers);

% Clean up auxiliary function variable.
clear tempLayers;
%%
% Connection layer branch.
% Connect all branches of the network to create a network lgraph.
lgraph = connectLayers(lgraph,"relu_1","conv_2");
lgraph = connectLayers(lgraph,"relu_1","addition_1/in2");
lgraph = connectLayers(lgraph,"relu_3","addition_1/in1");
lgraph = connectLayers(lgraph,"addition_1","maxpool_1");
lgraph = connectLayers(lgraph,"addition_1","addition_17/in1");
lgraph = connectLayers(lgraph,"relu_4","conv_5");
lgraph = connectLayers(lgraph,"relu_4","addition_2/in2");
lgraph = connectLayers(lgraph,"relu_6","addition_2/in1");
lgraph = connectLayers(lgraph,"addition_2","maxpool_2");
lgraph = connectLayers(lgraph,"addition_2","addition_16/in2");
lgraph = connectLayers(lgraph,"relu_7","conv_8");
lgraph = connectLayers(lgraph,"relu_7","addition_3/in2");
lgraph = connectLayers(lgraph,"relu_9","addition_3/in1");
lgraph = connectLayers(lgraph,"addition_3","maxpool_3");
lgraph = connectLayers(lgraph,"addition_3","addition_14/in2");
lgraph = connectLayers(lgraph,"relu_10","conv_11");
lgraph = connectLayers(lgraph,"relu_10","addition_4/in2");
lgraph = connectLayers(lgraph,"relu_12","addition_4/in1");
lgraph = connectLayers(lgraph,"addition_4","conv_13");
lgraph = connectLayers(lgraph,"addition_4","addition_12/in2");
lgraph = connectLayers(lgraph,"relu_13","conv_14");
lgraph = connectLayers(lgraph,"relu_13","addition_5/in2");
lgraph = connectLayers(lgraph,"relu_15","addition_5/in1");
lgraph = connectLayers(lgraph,"addition_5","conv_16");
lgraph = connectLayers(lgraph,"addition_5","addition_10/in2");
lgraph = connectLayers(lgraph,"relu_16","conv_17");
lgraph = connectLayers(lgraph,"relu_16","addition_6/in2");
lgraph = connectLayers(lgraph,"relu_18","addition_6/in1");
lgraph = connectLayers(lgraph,"addition_6","conv_19");
lgraph = connectLayers(lgraph,"addition_6","addition_9/in2");
lgraph = connectLayers(lgraph,"relu_19","conv_20");
lgraph = connectLayers(lgraph,"relu_19","addition_7/in2");
lgraph = connectLayers(lgraph,"relu_21","addition_7/in1");
lgraph = connectLayers(lgraph,"addition_7","addition_9/in1");
lgraph = connectLayers(lgraph,"addition_9","conv_42");
lgraph = connectLayers(lgraph,"addition_9","conv_22");
lgraph = connectLayers(lgraph,"transposed-conv_8","depthcat/in6");
lgraph = connectLayers(lgraph,"relu_22","conv_23");
lgraph = connectLayers(lgraph,"relu_22","addition_8/in2");
lgraph = connectLayers(lgraph,"relu_24","addition_8/in1");
lgraph = connectLayers(lgraph,"addition_8","addition_10/in1");
lgraph = connectLayers(lgraph,"addition_10","conv_25");
lgraph = connectLayers(lgraph,"addition_10","conv_41");
lgraph = connectLayers(lgraph,"relu_25","conv_26");
lgraph = connectLayers(lgraph,"relu_25","addition_11/in2");
lgraph = connectLayers(lgraph,"relu_27","addition_11/in1");
lgraph = connectLayers(lgraph,"addition_11","addition_12/in1");
lgraph = connectLayers(lgraph,"addition_12","transposed-conv_1");
lgraph = connectLayers(lgraph,"addition_12","conv_40");
lgraph = connectLayers(lgraph,"transposed-conv_6","depthcat/in4");
lgraph = connectLayers(lgraph,"relu_28","conv_29");
lgraph = connectLayers(lgraph,"relu_28","addition_13/in2");
lgraph = connectLayers(lgraph,"relu_30","addition_13/in1");
lgraph = connectLayers(lgraph,"addition_13","addition_14/in1");
lgraph = connectLayers(lgraph,"addition_14","conv_39");
lgraph = connectLayers(lgraph,"addition_14","transposed-conv_2");
lgraph = connectLayers(lgraph,"relu_31","conv_32");
lgraph = connectLayers(lgraph,"relu_31","addition_15/in2");
lgraph = connectLayers(lgraph,"relu_33","addition_15/in1");
lgraph = connectLayers(lgraph,"addition_15","addition_16/in1");
lgraph = connectLayers(lgraph,"transposed-conv_5","depthcat/in3");
lgraph = connectLayers(lgraph,"addition_16","transposed-conv_3");
lgraph = connectLayers(lgraph,"addition_16","conv_38");
lgraph = connectLayers(lgraph,"relu_34","conv_35");
lgraph = connectLayers(lgraph,"relu_34","addition_18/in2");
lgraph = connectLayers(lgraph,"transposed-conv_4","depthcat/in2");
lgraph = connectLayers(lgraph,"relu_36","addition_18/in1");
lgraph = connectLayers(lgraph,"addition_18","addition_17/in2");
lgraph = connectLayers(lgraph,"relu_37","depthcat/in1");
lgraph = connectLayers(lgraph,"transposed-conv_7","depthcat/in5");
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
% Training parameters
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',1, ...
    'Shuffle','every-epoch',...
    'ValidationData',ValidationData,...
    'Plots','training-progress',...
    'ExecutionEnvironment','cpu',...
    'CheckpointPath',checkpointpath);
%%
% Model training
[net,log] = trainNetwork(trainingData,lgraph,options);
end

