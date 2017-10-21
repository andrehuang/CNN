class net:
    def __init__(self):
        self.layers = []

    def load_layers:
        #will define how to create each layers

#in each layer it will have a element called class
class layer:
    def __init__(self):
        self.myclass = 1
        self.type = 'conv_mask'
        self.sliceMag = np.array([])

    def create_sliceMag(self,res_size,labelNum):
        self.sliceMag = np.zeros([res_size,labelNum]);
        
class residuls:
    def __init__(self):
        self.res = []

class res:
    def __init__(self):
        self.x = np.array([2,2,2])

layer_counts = len(net.layers)
layers = net.layers
end_layer = layers[layer_counts-1]
class_shape = end_layer.myclass.shape
labelNum = class_shape[2]

for lay in range(0, layer_counts):
    if net.layers[lay].type == 'conv_mask': # layer type is conv_mask
        if net.layers[lay].sliceMag.size == 0: # sliceMag size is empty
            res_size = residuls.res[lay].x.shape[2]
            net.layer[lay].create_sliceMag(res_size,labelNum)

        for lab in range (0, labelNum):
            #tmp = gather(max(max(res(lay).(:,:,:,net.layers{end}.class(:,:,lab,:) == 1),[],1),[],2))l
            if tmp.shape != 0:
                menatmp=mean(tmp[:,:,:1])

                if(sum(net.layers[lay].sliceMag[:,lab]) == 0):
                    net.layers[lay].sliceMag[:,lab] = max(meantmp,0.1);
                else:
                    tmptmp=0.9
                    meantmp(meantmp == 0)=net.layers[lay].sliceMag(meantmp==0);
                    net.layers[lay].sliceMag[:,lab]=np.multiply(net.layers[lay].sliceMag[:,lab],(tmptmp+meantmp(:)*(1-tmptmp)))
    
