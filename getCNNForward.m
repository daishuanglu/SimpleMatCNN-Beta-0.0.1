clear;close all;
%I=imread('car.jpg');
load('./etr/000032.mat')
eimg(64,:)=0;eimg(:,64)=0;
y=eimg(:);
clear fx1;
I=imread('000032.jpg');
img=imresize(I,[256,256]);
[height,len,width]=size(img);
%{
pwinlen=3; nflts=4;asmooth=0.2;
w{1}=rand(3,3,nflts); stride(1)=1;acttype(1)={'relu'};
w{2}=ones(pwinlen,pwinlen);stride(2)=2;acttype(2)={'maxpool'};
height=length(1:stride(2):height); len=length(1:stride(2):len);
w{3}=rand(3,3,nflts);stride(3)=1;acttype(3)={'relu'};
w{4}=ones(pwinlen,pwinlen);stride(4)=2;acttype(4)={'maxpool'};
height=length(1:stride(4):height); len=length(1:stride(4):len);
w{5}=rand(height*len*nflts,height*len);acttype(5)={'logit'};
w{6}=rand(height*len,height*len);acttype(6)={'logit'};
w{7}=rand(height*len,height*len);acttype(7)={'logit'};
%}
pwinlen=3; nflts=2;asmooth=0.2;
w{1}=rand(3,3,nflts); stride(1)=1;acttype(1)={'relu'};
w{2}=ones(pwinlen,pwinlen);stride(2)=4;acttype(2)={'maxpool'};
height=length(1:stride(2):height); len=length(1:stride(2):len);
w{3}=rand(height*len*nflts,height*len);acttype(3)={'logit'};

[hid,~] = bfnetforward(double(img),w,stride,asmooth, acttype);
w = bfnetbackprop(double(img),y,hid,w,stride,acttype,0.1);
[hid,out] = bfnetforward(double(img),w,stride,asmooth, acttype);

figure;
for i=1:2
subplot(1,2,i);imagesc(hid{2}(:,:,i));colormap gray;axis off;
end
axes('Position',[0 0 1 1],'Visible','off');
text(0.45,0.95,'poolConvtxt1','fontsize',12);
%{
figure;
for i=1:4
subplot(2,2,i);imagesc(hid{4}(:,:,i));colormap gray;axis off;
end
axes('Position',[0 0 1 1],'Visible','off');
text(0.45,0.95,'poolConvtxt2','fontsize',12);
%}
figure; imagesc(out);colormap gray;axis off;title('Output');
