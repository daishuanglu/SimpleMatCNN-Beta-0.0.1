function [hid,out] = bfnetforward(x,ws,strides,asmooth, acttype)
hid=cell(1,length(ws));
for ilayer=1:length(ws)
if ilayer==1, hprev=x; else hprev=hid{ilayer-1};end;
w=ws{ilayer};
try stride=strides(ilayer);catch stride=[];end;
% stride value types: 
%[] - FC layers; 1 - convolutional layers; >1 - pooling layers
[winlen,~,nflts]=size(w);
[height,len,width]=size(hprev);
%if nflts==1 && winlen==height*len*width && isempty(stride) % is fc layers?
if isempty(stride)
    fprintf('Forwarding FC layer %d. \n',ilayer);
    hid{ilayer}=activation(hprev(:)'*w,asmooth,acttype(ilayer));
    continue;
end
%[posx,posy]=find(hprev(:,:,1)+1);pos=[posx,posy];
st{1}=1:stride:height;st{2}=1:stride:len;
fx=zeros(length(st{1}),length(st{2}),nflts);
if stride==1,layertype='convolutional';else layertype='pooling';end;
for i=1:length(st{1})
    if mod(i,10)==0
        fprintf('Forwarding %s layer %d %.2f%% \n',layertype,ilayer,i/length(st{1})*100);
    end
    for j=1:length(st{2})
        if st{1}(i)+winlen-1>height, endpt1=height; else endpt1=(st{1}(i)+winlen-1);end;
        if st{2}(j)+winlen-1>len, endpt2=len; else endpt2=(st{2}(j)+winlen-1);end;
        %tt1=st{1}(i):(st{1}(i)+winlen-1);
        %tt2=st{2}(j):(st{2}(j)+winlen-1);
        tt1=st{1}(i):endpt1; tt2=st{2}(j):endpt2;
        [p,q]=meshgrid(tt1,tt2);
        bfx=sparse(p(:),q(:),1,height,len);
        bfx=bfx(:);
        [p0,q0]=meshgrid(st{1}(i):(st{1}(i)+winlen-1),st{2}(j):(st{2}(j)+winlen-1));
        wlnks=zeros(1,winlen^2);
        wlnks((p0(:)<=height)+(q0(:)<=len)==2)=1;
        if strcmp(layertype,'pooling')
            for k=1:width
                xx=hprev(:,:,k);
                xv=xx(:);
                fx(i,j,k)=activation(xv(bfx==1),asmooth,acttype(ilayer));
            end
        else
            for k=1:nflts
                ww=w(:,:,k);ww=ww(:);
                wv=spfun(@(spx) ww(wlnks==1),bfx);
                wv=repmat(wv(:),width,1);
                fx(i,j,k)=activation(hprev(:)'*wv,asmooth,acttype(ilayer));
            end
        end
    end
end
hid{ilayer}=fx;
end
out=softmax(hid{end});

end

function f=activation(xw,smooth,type)
   if strcmp(type,'logit')
        f=1./(1+exp(-smooth.*xw));
   end
   if strcmp(type,'relu')
        f=max(0,xw);
   end
    if strcmp(type,'maxpool')
        f=max(xw);
    end
    if strcmp(type,'minpool')
        f=min(xw);
    end
    if strcmp(type,'meanpool')
        f=mean(xw);
    end
end

function idx=vecismember(a,b)
dim=size(a,2);
tmp=ismember(a(:,1),b(:,1));
    for i=2:dim
        tmp=tmp+ismember(a(:,i),b(:,i));
    end
    idx=(tmp==dim);
end

function fx=softmax(x)
fx=exp(x)./sum(exp(x));
end