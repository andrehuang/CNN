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


def getMask(inputs):
    '''
    :param inputs:  a numpy array (When wrapped in a tensorflow function, inputs is a Tensor)
    :return: mask, which is a tensor corresponding to that tensor
    '''
    # global filter_type   # should be defined in our_cnn_mnist.py file
    shape = inputs.shape
    batchS, h, w, depth = shape

    def getMu(inpu=inputs):
        '''
        :param input: tensor , the layer
        :return: mu_x, mu_y; (mu_x, mu_y) is the location of the mu;
                the shapes of mu_x, mu_y are both [batchS, 1, 1, depth]
        '''

        nonlocal batchS, h, w, depth, shape

        emp = np.zeros([batchS, h, w, depth])
        tmp_x = emp + np.array(np.reshape(np.arange(1, h+1), newshape=(1, h, 1, 1)))
        tmp_y = emp + np.array(np.reshape(np.arange(1, w + 1), newshape=(1, 1, w, 1)))
        # Now both tmp_x, tmp_y both have shape (batchS, h, w, depth)

        sumX = np.maximum(np.sum(inpu, axis=(1, 2)), 0.000000001)
        # shape of sumX: [batchS, depth]
        tx = np.sum(np.multiply(tmp_x, inpu), (1, 2))
        ty = np.sum(np.multiply(tmp_y, inpu), (1, 2))
        # shape of tx, ty: [batchS, depth]
        mu_x = np.maximum(np.round(np.divide(tx, sumX)), 1)
        mu_y = np.maximum(np.round(np.divide(ty, sumX)), 1)
        # shape of mu_x, mu_y: [batchS, depth]

        # 想要以mu_x, mu_y的各个值为index来取出tmp的元素
        # tmp = np.linspace(-1, 1, h)
        # 肯定可以搞，但暂时没想出怎么用array indexing；不过发现可以直接数学地解决它

        # rescaling
        # Notice that mu_x and mu_y are integers in the range [1, h]
        mu_x = -1 + mu_x * 2/h
        mu_y = -1 + mu_y * 2/h
        # shape of mu_x, mu_y: [batchS, depth]
        mu_x = mu_x.reshape((batchS, 1, 1, depth))
        mu_y = mu_y.reshape((batchS, 1, 1, depth))
        return mu_x, mu_y

    mu_x, mu_y = getMu(inputs)
    # mu_x, mu_y 's shapes are both (batchS, 1, 1, depth)

    # 设一个index作为全局变量，每一层存一个对应的posTemp
    lx = np.linspace(-1, 1, w)
    lx = np.reshape(lx, newshape=(1, 1, w, 1))
    ly = np.transpose(np.linspace(-1, 1, h))
    ly = np.reshape(ly, newshape=(1, h, 1, 1))
    empTemp = np.zeros((batchS, h, w, depth))
    posTempX = empTemp + lx
    posTempY = empTemp + ly

    mu_x = empTemp + mu_x
    mu_y = empTemp + mu_y
    # mu_x, mu_y 's shapes are both [batchS, 1, 1, depth]
    mask = np.absolute(posTempX - mu_x)
    mask = np.add(mask, np.absolute(posTempY - mu_y))
    mask = np.maximum(1 - np.multiply(mask, 2), -1)

    # 这些细节需要和学长一起修改：
    # partRate。在大的层面里设置一个与filter_type(partRate)。固定的参数。
    # filter_type应该要被加在our_mnist里面，代表是否需要mask。

    # mask[:, :, filter_type != 1, :] = 1
    # print(mask)
    # print(mask)  # 看feature map
    return mask


def np_mask(x):
    """
    :param x: an array
    :return: an array
    """
    mask = getMask(x)
    # print(mask)
    x_new = np.multiply(mask, x)
    return np.maximum(x_new, 0).astype(np.float32)

# Test:
# test = np.arange(1, 61)
# test = np.reshape(test, [3, 2, 2, 5])
# np_mask(test)


# make the date type compatible with tensorflow
# np_mask_32 = lambda x: np_mask(x).astype(np.float32)


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


d_relu = np.vectorize(d_relu)  # vectorizing: making it into a numpy function
d_relu_32 = lambda x: d_relu(x).astype(np.float32)  # make data type compatible


# transform the numpy function into a Tensorflow function
def tf_d_relu(x, name=None):
    """
    :param x: a list of tensors (but we can just see it as a tensor)
    :param name: 
    :return: a tensor
    """
    with ops.name_scope(name, "d_relu", [x]) as name:  # where bug is
        z = tf.py_func(d_relu_32,
                       [x],
                       tf.float32,
                       name=name,
                       stateful=False)
        return z[0]


# tf.py_func acts on lists of tensors (and returns a list of tensors),
# that is why we have [x] (and return z[0]).

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
    return tf.multiply(grad, n_gr)


# our final op
def tf_mask(x, name=None):
    with ops.name_scope(name, "Mask", [x]) as name:
        z = py_func(np_mask,
                    [x],
                    [tf.float32],
                    name=name,
                    grad=our_grad)

        z = z[0]  # z is a tensor now
        # z.eval()
        # Since py_func can execute arbitrary Python code and output anything, TensorFlow can't figure out the shape
        # We will always have to manually set z's shape
        z.set_shape(x.get_shape())  # We need to give z a shape, in our context, it has the same shape as x
        # z = x
        return z

# Test:

# with tf.Session() as sess:
#             x = tf.constant([1., 2., 0.3, 0.1, -1., -1.5, 1.0, -8.0])
#             # x = tf.constant([[[1., 2., -3., 5.]], [[7., -8., 9., -10.]]])
#             x = tf.reshape(x, [2, 2, 2, 1])
#             x.eval()
#             # z = x
#             z = tf_mask(x)
#             # z.eval()
#             gr = tf.gradients(z, [x])[0]
#             tf.global_variables_initializer().run()
#             print(x.eval(), z.eval(), gr.eval())

# 几个改写的注意点：
# 1 Python中的index从0开始
# 2 关于shape, MATLAB中采用[h, w, depth, batchS]
#   python中采用[batchS, h, w, depth]
#   相应的sum的axis也要调整(from trial, np.sum(z,1) is equivalent to sum(z, 2) in MATLAB
# 3 np.arange(1, h+1) 与 MATLAB中(1:h)是一样的，这也是python的特点
# 4 shape的数字应该还是没问题的，实验了一下，[3,3]就是3*3
# 5 延续上一点，与shape相关的index都是natural的理解方式，比如还有在linspace中给出的参数。
