function [index] = Patch_to_num(patch,inputimage,inputlabel,Outputimagedir,Outputlabeldir,index)
%��.tif��ʽ������ͼ�����ǩͼ�����ָ����С��Ƭ

%   Author: Cheng Xin, Ocean University of China, Email: chengxin@stu.ouc.edu.cn

%   patch:��Ƭ��С
%   inputimage:����ͼ��·��
%   inputlabel:������Ĥ��ǩ·��
%   Outputimagedir:���ͼ��·��
%   Outputlabeldir:�����Ĥ��ǩ·��
%   index:��ʼ���ͼ�������

%   loadtiff()��������matlab���ӹ���������
A=loadtiff(inputimage);
B=loadtiff(inputlabel);
C=size(A);
C2=size(B);
D=size(size(B));
if(C(3)==4 && D(2)==2 && C(1)==C2(1) && C(2)==C2(2))
    A=uint8(A);
    B=uint8(B);
    row=C(1);
    col=C(2);
    if(mod(row,patch))
        row=ceil(row/patch)*patch;
    end
    if(mod(col,patch))
        col=ceil(col/patch)*patch;
    end
    rgbn=uint8(zeros([row,col,4]));
    rgbn(1:C(1),1:C(2),:)=A;
    rgb=rgbn(:,:,[3,2,1]);
    ngb=rgbn(:,:,[4,2,1]);
    mask=uint8(zeros([row,col]));
    mask(1:C(1),1:C(2))=B;
    for i=1:(row/patch)
        for j=1:(col/patch)
            Outputrgb=rgb(((i-1)*patch+1):i*patch,((j-1)*patch+1):j*patch,:);
            Outputngb=ngb(((i-1)*patch+1):i*patch,((j-1)*patch+1):j*patch,:);
            Outputlabel=mask(((i-1)*patch+1):i*patch,((j-1)*patch+1):j*patch);
            imwrite(Outputrgb,[Outputimagedir,num2str(index),'.png']);
            imwrite(Outputlabel,[Outputlabeldir,num2str(index),'.png']);
            imwrite(Outputngb,[Outputimagedir,num2str(index),'_ngb','.png']);
            imwrite(Outputlabel,[Outputlabeldir,num2str(index),'_ngb','.png']);
            index = index + 1;
        end
    end
    disp('SUCCESSFUL!&&Channel=4')
else if(C(3)==3 && D(2)==2 && C(1)==C2(1) && C(2)==C2(2))
        A=uint8(A);
        B=uint8(B);
        row=C(1);
        col=C(2);
        if(mod(row,patch))
            row=ceil(row/patch)*patch;
        end
        if(mod(col,patch))
            col=ceil(col/patch)*patch;
        end
        rgb=uint8(zeros([row,col,3]));
        rgb(1:C(1),1:C(2),:)=A;
        mask=uint8(zeros([row,col]));
        mask(1:C(1),1:C(2))=B;
        for i=1:(row/patch)
            for j=1:(col/patch)
                Outputrgb=rgb(((i-1)*patch+1):i*patch,((j-1)*patch+1):j*patch,:);
                Outputlabel=mask(((i-1)*patch+1):i*patch,((j-1)*patch+1):j*patch);
                imwrite(Outputrgb,[Outputimagedir,num2str(index),'.png']);
                imwrite(Outputlabel,[Outputlabeldir,num2str(index),'.png']);
                index = index + 1;
            end
        end
        disp('SUCCESSFUL!&&Channel=3')
    else
        disp('ERROR FORMAT!!!')
    end
    
end
end


