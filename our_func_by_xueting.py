# /usr/bin/python
# In tensorflow,
# As long as you're just composing ops that have gradients,
# the automatic differentiation will work.

# Right now we are attempting to add a new op,
# therefore we have to define the corresponding bprop function
# to get the gradients

# Here is a demo to try to define an relu op
# experimented in our_cnn_mnist
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from mask_cnn_mnist2 import getMask

class layer: # this is an example of one layer, should be replaced by input argument later; This is equivalent to l in matlab
    def __init__(self):
        self.filters = np.arange(4)
        self.sliceMag = np.random.random_integers(5, size=(5,5))
        self.mag = 0.1                     #TODO: value needs to be changed later


    def get_filters(self):
        return self.filters

    def set_strength(self, strength):
        self.strength = strength
  
# This class passes information to network  
class end_layer: 
    def __init__(self):
        self.iters = 0
        self.theclass = np.random.random_integers(5,size=(5,5,1,5)) #labelNum > 1
        # shape [h, w, labelNum, imageNum]
        # labelNum defines the number of categories

class div: #initialize Div struct by depthList and posList
    def __init__(self, depthList, posList):
        self.depthList = depthList
        self.posList = posList

alpha = 0.5
myLayer = layer()
endLayer = end_layer()
theFilters = myLayer.filters
endClass = endLayer.theclass
h = 0        
w = 0     
depth = 0     
batchS = 0  
mask = np.zeros([batchS, h, w, depth])  

def setup_div(layer, end_layer):
    depthList = np.nonzero(layer.filters>0)[0]
    labelNum = (end_layer.theclass.shape)[2]

    div_list = []
    # binary classificaiton
    if labelNum == 1:
        theClass = end_layer.theclass
        theClass = theClass.flatten('C')
        posList = np.nonzero(theClass==1)[0]
        div_list.append(div(depthList,posList))

    # multi-class classification
    else:
        theClass = np.amax(end_layer.theclass, axis = 2)
        theClass = theClass.flatten('C')
        posList = np.nonzero(theClass==1)[0]

        if layer.sliceMag.size == 0:
            div_list.append(div(depthList,posList))
        else:
            sliceM = layer.sliceMag
            idx = np.argmax(sliceM[depthList,:], axis = 1)
            for lab in range (0, labelNum):
                depthList = np.sort(depthList[idx==lab]) #这里我不是很懂怎么用depthList[idx == lab])
                theEndClass = (end_layer.theclass)[:,:,lab,:1]
                theEndClass = theEndClass.flatten('C')
                posList = np.nonzero(theEndClass==1)[0] #matlab find function 我不是很确定，因为它会把matrix转化成arrary, 所以我先用flatten了
                div_list.append(div(depthList, posList))
    return div_list


def setup_logz(layer,end_layer, grad, theInput):
    imgNum = (end_layer.theclass.shape)[3]
    
    grad = np.multiply(grad, np.maximum(mask, 0))

    # assume layer.filters == 1 since the situation when layer.filters == 2 is not implemented this time
    strength = np.reshape(np.mean(np.multiply(theInput, mask), axis = (1,2)), (batchS, depth))

    layer.set_strength(strength)
    alpha_logZ_pos = np.reshape(np.multiply(np.log(np.mean(
                        np.exp(np.divide(np.mean(np.multiply(theInput, mask[::-1,:,:,::-1]),axis= (1,2)), alpha)),axis = 0)), alpha), (1, depth))
    
    alpha_logZ_neg = np.reshape(np.multiply(np.log(np.mean(np.exp(np.divide(np.mean(-theInput, axis = (1,2)), alpha)), axis = 0)), alpha), (depth, 1))
    
    alpha_logZ_pos[np.isinf(alpha_logZ_pos)] = np.amax(alpha_logZ_pos[np.isinf(alpha_logZ_pos)==0])
    alpha_logZ_neg[np.isinf(alpha_logZ_neg)] = np.amax(alpha_logZ_neg[np.isinf(alpha_logZ_neg)==0])

    return alpha_logZ_pos, alpha_logZ_neg, grad



def post_process_gradient(grad, theInput, alpha_logZ_pos, alpha_logZ_neg, div_list):
    w_pos = 0.5
    w_neg = 0.5
    endLayer.iters += 1      # should this be incremented?

    for lab in range (0, len(div_list)):
        if len(div_list) == 1:
            w_pos = 1
            w_neg = 1
        else:
            density = endLayer.density            # shape ?
            w_pos = np.divide(0.5, density[lab])
            w_neg = np.divide(0.5, 1-density[lab])

        ## For parts
        imgNum = (endLayer.theclass.shape)[3]
        mag = np.divide(np.divide(np.ones([depth, imgNum]),np.divide(1, endLayer.iters)),myLayer.mag)      # iter, mag ?
        dList = div_list[lab].depthList
        dList = dList[myLayer.filters[dList]==1]
        if not dList.size > 0:
            theList = div_list[lab].posList
            if not theList.size > 0:
                strength = np.multiply(np.exp(np.divide(myLayer.strength[dList,theList], alpha)),myLayer.strength[dList,theList] - np.tile(alpha_logZ_pos[dlist], [1, theList.size])+alpha)
                strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength)==0])
                strength[np.isnan(strength)] = 0
                strength = np.reshape(np.multiply(np.divide(strength, np.tile(np.mean(strength, 1), [1, theList.size])), mag[dList, theList]), [1, 1, dList.size, theList.size])
                strength[np.isnan(strength)] = 0
                strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength)==0])

                updated_value = -np.multiply(np.multiply(-mask[:,:,dList,theList],np.tile(strength, [h,w,1,1])),0.00001 * w_pos)
                grad[:,:,dList, theList] += updated_value

            theList_neg = np.setdiff1d(np.arange(batchS), div_list[lab].posList)
            if not theList_neg.size > 0:
                strength = np.reshape(np.mean(np.mean(theInput[:,:,dList,theList_neg],axis = 0),axis = 1), [dList.size, theList_neg.size])
                strength = np.multiply(np.exp(np.divide(-strength,alpha)), (-strength-np.tile(alpha_logZ_neg[dList], [1, theList_neg.size]) + alpha))
                strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength)==0])
                strength[np.isnan(strength)] = 0
                strength = np.reshape(np.divide(strength, np.tile(np.mean(strength, axis=1), [1, list_neg.size])*mag(dList, theList_neg)), [1,1,dList.size, theList_neg.size])
                strength[np.isnan(strength)] = 0
                strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength)==0])
                updated_value_neg = np.multiply(np.tile(np.reshape(strength, [1,1,dList.size, theList_neg.size]), [h,w,1,1]), (0.00001*w_neg))
                grad[:,:,dlist, theList_neg] += updated_value_neg
    
    return grad


def process_gradient(x, grad):
    print("In process_gradient")

    div_list = setup_div(myLayer, endLayer)
    
    global h, w, depth, batchS, mask
    mask = getMask(x)
    batchS = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    depth = x.shape[3]

    alpha_logZ_pos, alpha_logZ_neg, grad = setup_logz(myLayer,endLayer, grad, x)
    grad = post_process_gradient(grad, x, alpha_logZ_pos, alpha_logZ_neg, div_list)
    return grad.astype(np.float32)

#Code above was added by XuetingYan

# our hack
def py_func(func, inp, tout, stateful=True, name=None, grad=None):
    """
    I omitted the introduction to parameters that are not of interest
    :param func: a numpy function
    :param inp: input tensors
    :param grad: a tensorflow function to get the gradients (used in bprop, should be able to receive previous 
                gradients and send gradients down.)
    
    :return: a tensorflow op with a registered bprop method —— This is what we WANT!
    """
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1000000))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, tout, stateful=stateful, name=name)


def np_relu(x):
    """
    :param x: an array
    :return: an array
    """
    return np.maximum(x, 0)
# make the date type compatible with tensorflow
np_relu_32 = lambda x: np_relu(x).astype(np.float32)


def tf_process_gradient(x, grad, name=None):
    print("getting up this function")
    with ops.name_scope(name, "process_gradient_name", [x, grad]) as name: 
        print("getting up to process_gradient")
        z = tf.py_func(process_gradient,[x, grad],[tf.float32],name=name,stateful=False)
        return z[0]


def our_grad(cus_op, grad):
    """Compute gradients of our custom operation.
    Args:
        param cus_op: our custom op
        param grad: the previous gradients before the operation
    Returns:
        gradient that can be sent down to next layer in back propagation
        it's an n-tuple, where n is the number of arguments of the operation   
        """
    x = cus_op.inputs[0]
    n_gr2 = tf_process_gradient(x, grad)
    return n_gr2


# our final op
def tf_relu(x, name=None):
    with ops.name_scope(name, "OurRelu", [x]) as name:   # where bug is
            z = py_func(np_relu_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=our_grad)
                    # print(z)  # [<tf.Tensor 'OurRelu:0' shape=<unknown> dtype=float32>]
            z = z[0]      # z is a tensor now
            z.set_shape(x.get_shape())    # We need to give z a shape
            return z  # Tensor("OurRelu:0", dtype=float32)


def main():

    myLayer = layer()
    endLayer = end_layer()
    theFilters = myLayer.filters
    endClass = endLayer.theclass


    div_list = setup_div(myLayer, endLayer)

    with tf.Session() as sess:
        x = tf.constant([[[[7., 9., -8.],[-3,4,-3]], [[4., -9, 7.], [3., -6., -1.]]]])
        x.eval()
        z = tf_relu(x)
        gr = tf.gradients(z, [x])[0]
        tf.global_variables_initializer().run()
        # print(x.eval())
        # print(z.eval())
        # print(z.eval())
        # print(gr.eval())


if __name__ == "__main__": 
    main()
