function [] = train_label(Pathrgbn,Pathmask,Outputimagedir,Outputlabeldir,patch,index)
%用于批量处理文件夹内训练图像和标签图像为指定大小的图像切片，需要调用Patch_to_num函数

%Author: Cheng Xin, Ocean University of China, Email: chengxin@stu.ouc.edu.cn

%Pathrgbn：训练图像文件夹路径
%Pathmask：标签图像文件夹路径
%Outputimagedir：输出训练图像文件夹路径
%Outputlabeldir：输出标签图像文件夹路径
%patch：切片大小
%index：输出图像名称起始索引,其为可选参数，默认为1
%注意：为了避免出错，请将训练图像与对应标签图像用相同名字命名


Filergbn = dir(fullfile(Pathrgbn,'*.tif'));
Filermask = dir(fullfile(Pathmask,'*.tif'));
FileNamesrgbn = {Filergbn.name};
FileNamesmask = {Filermask.name};
filenum=size(FileNamesrgbn);

if ~exist(Outputimagedir,'file')
    mkdir(Outputimagedir);
end
if ~exist(Outputlabeldir,'file')
    mkdir(Outputlabeldir);
end



if(nargin==5)
    index=1;
    for i=1:filenum(2)
        if(strcmp(FileNamesmask{i},FileNamesrgbn{i}))
            inputimage=[Pathrgbn,FileNamesrgbn{i}];
            inputlabel=[Pathmask,FileNamesmask{i}];
            disp(['Processing ','No.',num2str(i),'/',num2str(filenum(2)),' Picture:',FileNamesrgbn{i}])
            index=Patch_to_num(patch,inputimage,inputlabel,Outputimagedir,Outputlabeldir,index);
        else
            disp(['系统将',FileNamesrgbn{i},'与',FileNamesmask{i},'对应。'])
            disp('图像与标签未对应，建议将命名统一。')
            break;
        end
    end
elseif(nargin==6)
    for i=1:filenum(2)
        if(strcmp(FileNamesmask{i},FileNamesrgbn{i}))
            inputimage=[Pathrgbn,FileNamesrgbn{i}];
            inputlabel=[Pathmask,FileNamesmask{i}];
            disp(['Processing ','No.',num2str(i),'/',num2str(filenum(2)),' Picture:',FileNamesrgbn{i}])
            index=Patch_to_num(patch,inputimage,inputlabel,Outputimagedir,Outputlabeldir,index);
            
        else
            disp(['系统将',FileNamesrgbn{i},'与',FileNamesmask{i},'对应。'])
            disp('图像与标签未对应，建议将命名统一。')
            break;
        end
    end
else
    disp('输入参数不正确。')
end
end


