#section 1

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

if l.type == 'conv_mask':
    posTempX=np.tile(l.posTemp[1],[1,1,1,res[i].shape[3]);
    posTempY=np.tile(l.posTemp[2],[1,1,1,res[i].shape[3]);

    res_shape = res[i].x.shape;
    h = res_shape[0];
    w = res_shape[1];
    depth = res_shape[2];
    batchS = res_shape[3];

    #[net.layers{i}.mu_x,net.layers{i}.mu_y,net.layers{i}.sqrtvar]=getMu(res(i).x);
    #mask=getMask(net.layers{i},h,w,batchS,depth,posTempX,posTempY);
    input=np.multiply(res[i].x,mask)


if l.type == 'conv_mask':

    posTempX=np.tile(l.posTemp[1],[1,1,1,res[i].shape[3]);
    posTempY=np.tile(l.posTemp[2],[1,1,1,res[i].shape[3]);

    res_shape = res[i].x.shape;
    h = res_shape[0];
    w = res_shape[1];
    depth = res_shape[2];
    batchS = res_shape[3];

    #mask=getMask(net.layers{i},h,w,batchS,depth,posTempX,posTempY);
    input=np.multiply(res[i].x,mask)

    l_filter = l.filter
    end = len(net.layers)-1
    depthList = indices(a, lambda x: x > 2)
    labelNum = net.layers[end].class.shape[2]

    if labelNum == 1:
        theClass=net.layer[end].class
        Div = struct('depthList',depthList,'posList',indics(theClass, lambda x: x == 1)) # will be defined later
    else:
        theClass=max(net.layers[end].class,3)
        if(l.sliceMag.shape == 0):
            Div = struct('depthList',depthList,'posList',indics(theClass, lambda x: x == 1))
        else:
            sliceM = l.sliceMag
            Div = np.tile(struct('depthList',[],'posList',[]),[1,labelNum])
            [a,idx]=sliceM(depthList).max(1)
            for lab in range(0,labelNum):
                Div[lab].depthList=sort(depthList[depthList(idx==lab)])
                Div[lab].posList=indics(net.layers[end].class[:,:,lab,:], lambda x: x == 1)

    imgNum=net.layers[end].class[3]

    alpha=0.5
    res[i].dzdx=np.multiply(res[i].dzdx,max(mask,0))

    if(sum(l.filter==1)>0):
        #mask_texture=getMask_texture(l,h,w,res(i).x);
        #resX=res(i).x./repmat(max(max(sum(sum(res(i).x,1),2),[],4),0.00001),[size(res(i).x,1),size(res(i).x,2),1,size(res(i).x,4)]).*15;
        #magX=reshape(mean(max(max(res(i).x,[],1),[],2),4),[depth,1])
        
         alpha_logZ_pospos=reshape(log(mean(exp(mean(mean(res(i).x.*mask_texture,1),2)./alpha),4)).*alpha,[depth,1]);
         alpha_logZ_pospos(isinf(alpha_logZ_pospos))=max(alpha_logZ_pospos(isinf(alpha_logZ_pospos)==0));
         alpha_logZ_negneg=reshape(log(mean(exp(mean(mean(-resX,1),2)./alpha),4)).*alpha,[depth,1]);
         alpha_logZ_negneg(isinf(alpha_logZ_negneg))=max(alpha_logZ_negneg(isinf(alpha_logZ_negneg)==0));


    for lab in range(0,Div.size):
        if Div.size == 1:
            w_pos = 1
            w_neg = 1
        else:
            w_pos = np.divide(0.5,net.layers[end].density[lab])
            w_neg = np.divide(0.5,1-net.layers[end].density[lab])

        mag = np.divide(np.divide(np.ones([depth,imgNum]),np.divide(1,net.layers[end].iter),l.mag))
        dList=Div[lab].depthList
        dList=dList[(l.filter[dList]==1)

        if dList.shape != 0:
            mylist = Div(lab).posList
            if mylist.shape != 0:
                mystrength = l.strength
                this_strength = mystrength[dList][mylist]
                strength = np.multiply(np.exp(np.divide(this_strength,alpha)),this_strength-np.tile(alpha_logZ_pos[dList],[1,myList.size])+alpha)
                strength[np.isinf(strength)] = np.max(strength[np.isinf(strength)==0])
                strength[np.isnan(strength)] = 0
                strength[np.isinf(strength)] = np.max(strength[isinf(strength)==0])
                # normalized by mean of strength:problems
                dzdx = res[i].dzdx
                dzdx[:,:,dList,mylist] = dzdx[:,:,dList,mylist]-np.multiply(mask[:,:,dList,mylist],np.tile(strength,[h,w,1,1]))*0.00001*w_pos
                
            batchArray = np.arange(batchSize)
            list_neg = np.setdiff1d(batchArray,Div[lab].posList)
            
            if list_neg.size != 0:
                strength=np.reshape(np.mean(np.mean(res[i].x[:,:,dList,mylist],axis=0),axis=1),[dList.size,list_neg.size])
                strength=np.multiply(np.exp(np.divide(-strength,alpha)),(-strength-np.repmat(alpha_logZ_neg[dList],[1,list_neg.size])+alpha))
                strength[np.isinf(strength)] = np.max(strength[np.isinf(strength)==0])
                strength[np.isnan(strength)] = 0
                strength = np.reshape(np.divide(strength,np.tile(np.mean(strength,axis=1),[1,list_neg.size])*mag(dList,list_neg)),[1,1,dList.size,list_neg.size])
                strength[np.isnan(strength)] = 0
                strength[np.isinf(strength)] = np.max(strength[np.isinf(strength)==0])
                dzdx = res[i].dzdx
                dzdx[:,:,dList,list_neg] = dzdx[:,:,dList,list_neg]+np.tile(np.reshape(strength,[1,1,dList.size,list_neg.size]),[h,w,1,1])*0.00001*w_neg 
