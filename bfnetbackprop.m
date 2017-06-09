function w = bfnetbackprop(x,y,hid,ws,strides,acttype,yida)
%[hid,~] = bfnetforward(x,ws,strides,asmooth, acttype);
nlys=length(ws);
%w=ws{end}; try stride=strides(end); catch stride=[];end;
%if length(hid)>1, hprev=hid{end-1}; else hprev=x; end;
%[winlen,~,nflts]=size(w);
beta=(y-hid{end}(:))';
for c=0:length(ws)-1
    ilayer=nlys-c; %n-1
    w=ws{ilayer}; try stride=strides(ilayer); catch stride=[];end;
    if ilayer>1, hprev=hid{ilayer-1}; else hprev=x; end;
    [winlen,~,nflts]=size(w);
    hv=hid{ilayer}(:)'.*(1-hid{ilayer}(:)');
    dc=length(hv);
    if isempty(stride)
        fprintf('FC layer %d backpp. \n',ilayer);
        for jc=1:dc
            w(:,jc)=w(:,jc)+yida*beta(jc)*hv(jc).*hprev(:);
        end
        switch acttype{ilayer}
            case 'relu', beta=sum(repmat(beta,size(w,1),2).*w,2)';
            case 'logit', beta=sum(repmat(beta,size(w,1),2),2)'.*hprev(:)'.*(1-hprev(:)');  
            otherwise, beta=sum(repmat(beta,size(w,1),2).*w,2)';
        end
    else
        if stride==1,layertype='Convolutional';else layertype='Pooling';end;
        w=reshape(w,[winlen^2,nflts]);
        [preh,prel,prew]=size(hprev);
        hprevv=reshape(hprev,[preh*prel,prew]);
        st{1}=1:stride:preh;st{2}=1:stride:prel;
        bfconng=sparse(preh*prel*prew,dc,0);
        for jc=1:dc
            if mod(jc,1000)==0
                fprintf('%s layer %d backpp %.2f%% \n',layertype,ilayer,jc/dc*100);
            end
            [ipre,jpre,kpre] = ind2sub(size(hid{ilayer}), jc);
            %jc-th nodes output is due to the filtering at (i,j)-th prev
            % with kpre-th filter w(:,:,k)
            if st{1}(ipre)+winlen-1>preh, endpt1=preh; else endpt1=(st{1}(ipre)+winlen-1);end;
            if st{2}(jpre)+winlen-1>prel, endpt2=prel; else endpt2=(st{2}(jpre)+winlen-1);end;
            tt1=st{1}(ipre):endpt1; tt2=st{2}(jpre):endpt2;
            [p,q]=meshgrid(tt1,tt2);
            bfx=sparse(p(:),q(:),1,preh,prel);
            bfx=bfx(:);
            wlnks=zeros(1,winlen^2);
            [p0,q0]=meshgrid(st{1}(ipre):(st{1}(ipre)+winlen-1),st{2}(jpre):(st{2}(jpre)+winlen-1));    
            wlnks((p0(:)<=preh)+(q0(:)<=prel)==2)=1;
            if stride>1
                idw=find(wlnks==1);
                bfhprev=hprevv(bfx==1,:);
                [bfval,idmax]=max(bfhprev,[],1);
                for iii=1:length(idmax)
                    w(idw(idmax(iii)))=w(idw(idmax(iii)))+yida*hv(jc).*bfval(iii).*beta(jc);
                end
                wvals=repmat(w(idw),1,prew);
            else
                bfhprev=hprevv(bfx==1,kpre);                
                w(wlnks==1,kpre)=w(wlnks==1,kpre)+yida*hv(jc).*bfhprev.*beta(jc);
                wvals=repmat(w(wlnks==1,kpre),prew,1);
            end
            
            %bfconng(sub2ind(size(bfconng),1:preh*prel*prew,jc.*ones(1,preh*prel*prew)))...
                    %=spfun(@(spx)wvals(:),repmat(bfx,prew,1));
                bfconng(:,jc)=spfun(@(spx)wvals(:),repmat(bfx,prew,1));
            
        end
        w=reshape(w,[winlen,winlen,nflts]);
        switch acttype{ilayer}
            case 'relu', beta=sum(bfconng,2)';
            case 'logit', beta=sum(bfconng,2)'.*hprev(:)'.*(1-hprev(:)');
            otherwise, beta=sum(bfconng,2)';
        end
    end
    ws{ilayer}=w;
    %beta=sum(repmat(beta,size(w{c},1),1).*w{c},2)'.*bfhprev.*(1-bfhprev);
    % beta needs same dimension of the c-1 layer
end
end


