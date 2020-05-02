function [] = train_label(Pathrgbn,Pathmask,Outputimagedir,Outputlabeldir,patch,index)
%�������������ļ�����ѵ��ͼ��ͱ�ǩͼ��Ϊָ����С��ͼ����Ƭ����Ҫ����Patch_to_num����

%Author: Cheng Xin, Ocean University of China, Email: chengxin@stu.ouc.edu.cn

%Pathrgbn��ѵ��ͼ���ļ���·��
%Pathmask����ǩͼ���ļ���·��
%Outputimagedir�����ѵ��ͼ���ļ���·��
%Outputlabeldir�������ǩͼ���ļ���·��
%patch����Ƭ��С
%index�����ͼ��������ʼ����,��Ϊ��ѡ������Ĭ��Ϊ1
%ע�⣺Ϊ�˱�������뽫ѵ��ͼ�����Ӧ��ǩͼ������ͬ��������


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
            disp(['ϵͳ��',FileNamesrgbn{i},'��',FileNamesmask{i},'��Ӧ��'])
            disp('ͼ�����ǩδ��Ӧ�����齫����ͳһ��')
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
            disp(['ϵͳ��',FileNamesrgbn{i},'��',FileNamesmask{i},'��Ӧ��'])
            disp('ͼ�����ǩδ��Ӧ�����齫����ͳһ��')
            break;
        end
    end
else
    disp('�����������ȷ��')
end
end


