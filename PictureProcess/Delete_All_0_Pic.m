function [index] = Delete_All_0_Pic(Trainpath,Labelpath)
%�˺�������ɾ����ü����µ�����ȫ��Ϊ0��ͼƬ�����ѵ��Ч��
%   Author: Cheng Xin, Ocean University of China, Email: chengxin@stu.ouc.edu.cn
%   TrainPath:ѵ��ͼ��·��
%   LabelPath:��ǩ·��
%   index:��¼ɾ�����ļ���

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

