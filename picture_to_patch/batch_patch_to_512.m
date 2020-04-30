clc;
clear;
%输入数据文件夹与输出路径
Pathrgbn = '';
Pathmask = '';
Outputimagedir='';
Outputlabeldir='';

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

index=1;
patch=512;
for i=1:filenum(2)
    inputimage=[Pathrgbn,FileNamesrgbn{i}];
    inputlabel=[Pathmask,FileNamesmask{i}];
    index=Patch_to_num(patch,inputimage,inputlabel,Outputimagedir,Outputlabeldir,index);
end