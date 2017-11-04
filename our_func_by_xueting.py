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

class layer: # this is an example of one layer, should be replaced by input argument later; This is equivalent to l in matlab
    def __init__(self):
        self.filters = np.arange(4)
        self.sliceMag = np.random.random_integers(10, size=(10,10))

    def get_filters(self):
        return self.filters
    
class end_layer: # this is a test for end layer which will pass class information to network
    def __init__(self):
        self.theclass = np.random.random_integers(5,size=(1,10,1,10))

class div: #initialize Div struct by depthList and posList
    def __init__(self, depthList, posList):
        self.depthList = depthList
        self.posList = posList

def setup_div(layer, end_layer):
    depthList = np.nonzero(layer.filters>1)[0]
    #print (depthList)
    
    labelNum = (end_layer.theclass.shape)[2]
    #print (labelNum)

    div_list = []
    if labelNum == 1:
        theClass = end_layer.theclass
        theClass = theClass.flatten('F')
        posList = np.nonzero(theClass==1)[0]
        div_list.append(div(depthList,posList))
    else:
        theClass = np.amax(end_layer.theclass, axis = 2)
        if layer.sliceMag.size == 0:
            div_list.append(depthList,posList)
        else:
            sliceM = layer.sliceMag
            idx = np.argmax(sliceM[depthList,:], axis = 1)
            for lab in range (0, labelNum):
                depthList = np.sort(depthList[idx==lab])
                theEndClass = (end_layer.theclass)[:,:,lab,:1]
                theEndClass = theEndClass.flatten('F')
                posList = np.nonzero(theEndClass==1)[0] #matlab find function 我不是很确定，因为它会把matrix转化成arrary, 所以我先用flatten了
                div_list.append(div(depthList, posList))
    
    return div_list
    #print (len(div_list))
    #print (div_list[0].depthList)
    #print (div_list[0].posList)

def setup_logz(layer,end_layer, grad, mask, theInput, depth, batchS):
    imgNum = (end_layer.theclass.shape)[3]

    alpha = 0.5
    grad = np.multiply(grad, np.maximum(mask,0))

 #I think we can assume layer.filters == 1 since the situation when layer.filters == 2 is not implemented this time
    strength = np.reshape(np.mean(np.mean(np.multiply(grad, mask), axis=0), axis=1), [depth, batchS])
    layer.set_strength(strength)
    alpha_logZ_pos = np.reshape(np.log(np.mean(np.exp(np.mean(np.mean(np.multiply(theInput, mask[:,:,end:1:-1,end:1:-1]),axis = 0),axis = 1)),axis = 3)) * alpha, [depth, 1])
    alpha_logZ_neg = np.reshape(np.log(np.mean(np.exp(np.mean(np.mean(-theInput, axis = 0), axis = 1)/0.5), axis = 3)) * 0.5, [depth, 1])
    alpha_logZ_pos[np.isinf(alpha_logZ_pos)] = np.max(alpha_logZ_pos[np.isinf(alpa_logZ_pos)==0])
    alpha_logZ_neg[np.isinf(alpha_logZ_neg)] = np.max(alpha_logZ_neg[np.isinf(alpa_logZ_neg)==0])

    return alpha_logZ_pos, alpha_logZ_neg

#    if np.sum(layer.filter==2) > 0: #a little complicated; todo 


def post_process_gradient(layer, end_layer, grad, alpha_logZ_pos, alpha_logZ_neg, div_list):
    for lab in range (0, len(div_list)):
        if len(div_list) == 1:
            w_pos = 1
            w_neg = 1
        else:
            density = end_layer.density
            w_pos = np.divide(0.5, density[lab])
            w_neg = np.divide(0.5, 1-density[lab])
        
        ## For parts
        mag = (np.divide(np.ones([depth, imgNum]),np.divide(1, end_layer.iters)),layer.mag)
        dList = div_list[lab].depthList
        dList = dList[layer.filters[dList]==1]
        if not np.isempty(dList):
            theList = div_list[lab].posList
            if not np.isempty(thelist):
                strength = np.multiply(np.exp(np.divide(layer.strength[dList,theList], alpha)),layer.strength[dList,theList] - np.tile(alpha_logZ_pos[dlist], [1, theList.size])+alpha)
                strength[np.isinf(strength)] = np.max(strength[np.isinf(strength)==0])
                strength[np.isnan(strength)] = 0
                strength = np.reshape(np.multiply(np.divide(strength, np.tile(np.mean(strength, 1), [1, theList.size])), mag(dList, theList)), [1,1, dList.size, theList.size])
                strength[np.isnan(strength)] = 0
                strength[np.isinf(strength)] = np.max(strength[np.isinf(strength)==0])

                updated_value = -np.multiply(np.multiply(-mask[:,:,dList,theList],np.tile(strength, [h,w,1,1])),0.00001 * w_pos)
                grad[:,:,dList, theList] += updated_value

            theList_neg = np.setdiff1d(np.arange(batchS), div_list[lab].posList)
            if not np.isempty(theList_neg):
                strength = np.reshape(np.mean(np.mean(grad[:,:,dList,theList_neg],0),1), [dList.size, theList_neg.size])
                strength = np.multiply(np.exp(np.divide(-strength,alpha)), (-strength-np.tile(alpha_logZ_neg[dList], [1, theList_neg.size]) + alpha))
                strength[np.isinf(strength)] = np.max(strength[np.isinf(strength)==0])
                strength[np.isnan(strength)] = 0
                strength = np.reshape(np.divide(strength, np.tile(np.mean(strength, axis=1), [1, list_neg.size])*mag(dList, theList_neg)), [1,1,dList.size, theList_neg])
                strength[np.isnan(strength)] = 0
                strength[np.isinf(strength)] = np.max(strength[np.isinf(strength)==0])
                updated_value_neg = np.multiply(np.tile(np.reshape(strength, [1,1,dList.size, theList_neg.size]), [h,w,1,1]), (0.00001*w_neg))
                grad[:,:,dlist, theList_neg] += updated_value_neg
    
    return grad

def process_gradient(x, grad):
    layer, end_layer = initialization() #这个function要改在外面declare
    div_list = set_div(layer, end_layer)
    mask = getMask(x)
    h = x.shape[0]
    w = x.shape[1]
    depth = x.shape[2]
    batchS = x.shape[3]
    alpha_logZ_pos, alpha_logZ_neg = setup_logz(layer,end_layer, grad, mask, x, depth, batchS)
    grad = post_process_gradient(layer, end_layer, grad, alpha_logZ_pos, alpha_logZ_neg, div_list)
    return grad

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


# The next three functions help us get a tf function to get gradients in bprop
# corresponding to our customized op

def d_relu(x):
    """
    CURRENTLY,
    :param x: a number
    :return:  a number
    BUT as long as we can define this function with array input and array output, i.e. a numpy function,
    We don't need to vectorize it later.
    """
    if x > 0:
        return 1
    else:
        return 0
d_relu = np.vectorize(d_relu) # vectorizing: making it into a numpy function
d_relu_32 = lambda x: d_relu(x).astype(np.float32)  # make data type compatible


# transform the numpy function into a Tensorflow function
def tf_d_relu(x, name=None):
    """
    :param x: a list of tensors (but we can just see it as a tensor)
    :param name: 
    :return: a tensor
    """
    with ops.name_scope(name, "d_relu", [x]) as name:   # where bug is
        z = tf.py_func(d_relu_32,
                    [x],
                    [tf.float32],
                    name=name,
                    stateful=False)
        return z[0]
# tf.py_func acts on lists of tensors (and returns a list of tensors),
# that is why we have [x] (and return z[0]).

def add_relu_32(x, row):
    x[row] += x[row]
    return x.astype(np.float32)

def tf_add_relu(x, name=None):

    with ops.name_scope(name, "d_relu", [x]) as name:
        z = tf.py_func(add_relu_32,
                    [x, 1],
                    [tf.float32],
                    name=name,
                    stateful=False)
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
    n_gr = tf_d_relu(x)
    n_gr = tf_add_relu(n_gr)
    our_grad = tf.multiply(grad, n_gr)
    our_grad = process_gradient(x, our_grad)
    return n_gr

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

# Test:
'''
with tf.Session() as sess:
    x = tf.constant([[[7., 9., -8.],[-3,4,-3]],[[4.,-9.,7.],[3.,-6.,-1.]]])
    x.eval()
    z = tf_relu(x)
#     # z.set_shape(x.get_shape())
    gr = tf.gradients(z, [x])[0]
    tf.global_variables_initializer().run()
    print(x.eval())
    print(z.eval())
    print(gr.eval())
'''
def main():
    myLayer = layer()
    endLayer = end_layer()
    theFilters = myLayer.filters
    endClass = endLayer.theclass

    print (theFilters)
    print (endClass)

    setup_div(myLayer, endLayer)

if __name__ == "__main__": 
    main()
