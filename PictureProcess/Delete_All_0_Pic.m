function [index] = Delete_All_0_Pic(Trainpath,Labelpath)
%此函数用于删除因裁剪导致的像素全部为0的图片，提高训练效率
%   Author: Cheng Xin, Ocean University of China, Email: chengxin@stu.ouc.edu.cn
%   TrainPath:训练图像路径
%   LabelPath:标签路径
%   index:记录删除的文件名

Traindata=dir(fullfile(Trainpath,'*.png'));
TraindataName={Traindata.name};
Labeldata=dir(fullfile(Labelpath,'*.png'));
LabeldataName={Labeldata.name};
num=size(TraindataName);
num=num(2);
index={};
for i=1:num
    Picturename=[Trainpath,TraindataName{i}];
    LabelName=[Labelpath,LabeldataName{i}];
    img=imread(Picturename);
    if(~sum(sum(sum(img))))
        index=[index,TraindataName{i}];
        delete(Picturename);
        delete(LabelName);
    end
    if(~mod(i,100))
        disp(['Processing ',num2str(i),'/',num2str(num)]);
    end
end
disp('Finished!');
end

