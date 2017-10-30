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

# posTemp
def getMask(inputs):
    '''
    :param inputs: should be a tensor (last layer's output)
    :return: mask, which is a tensor corresponding to that tensor
    '''
    global filter_type   # should be defined in our_cnn_mnist.py file

    shape = inputs.get_shape().as_list()  # [batch_size, 7, 7, 64] in mnist example
    batchS, h, w, depth = shape

    # checked
    def getMu(input=inputs):
        '''
        :param input: tensor , the layer
        :return: mu_x, mu_y, sqrtvar; (mu_x, mu_y) is the location of the 
        '''
        # 几个注意点：
        # 1 Python中的index从0开始
        # 2 关于shape, MATLAB中采用[h, w, depth, batchS]
        #   python中采用[batchS, h, w, depth]
        #   相应的sum的axis也要调整(from trial, np.sum(z,1) is equivalent to sum(z, 2) in MATLAB
        # 3 np.arange(1, h+1) 与 MATLAB中(1:h)是一样的，这也是python的特点
        # 4 shape的数字应该还是没问题的，实验了一下，[3,3]就是3*3
        # 5 延续上一点，与shape相关的index都是natural的理解方式，比如还有在np.tile中给出的参数。
        nonlocal batchS, h, w, depth, shape
        # 下面是对MATLAb代码的修改
        # flatten depth-wise
        layer = tf.reshape(input, [batchS, h*w, depth])
        # constructs templates
        tmp_x = np.array(np.tile(np.transpose(np.arange(1, h+1)), [batchS, w, depth]))
        tmp_y = np.array(np.reshape(np.tile(np.arange(1, w+1), [batchS, h, 1, depth]), [batchS, h * w, depth]))

        sumX = np.maximum(np.sum(layer, 0), 0.000000001)

        mu_x = np.maximum(np.round(np.divide(np.sum(np.multiply(tmp_x, layer), 0), sumX)), 1)
        mu_y = np.maximum(np.round(np.divide(np.sum(np.multiply(tmp_y, layer), 0), sumX)), 1)
        sqrtvar = np.sqrt(
                    np.divide((np.sum(np.multiply(np.square(tmp_x - np.tile(mu_x, [1, h * w, 1])), layer), 0) +
                           np.sum(np.multiply(np.square(tmp_y - np.tile(mu_y, [1, h * w, 1])), layer), 0)), sumX))

        np.delete(arr=[sumX, tmp_x, tmp_y])

        # It seems the next two lines of codes are not used, I commented them
        # [maxX, ~] = max(x, [], 1)
        # p = np.tile(np.mutilply(mu_x + (mu_y - 1),h), [batchS, 1, 1, depth])

        tmp = np.linspace(-1, 1, h)
        mu_x = np.tile(tmp[mu_x], [batchS, 1, 1, depth])
        mu_y = np.tile(tmp[mu_y], [batchS, 1, 1, depth])
        sqrtvar = np.tile(sqrtvar, [batchS, depth])
        return mu_x, mu_y, sqrtvar

    mu_x, mu_y, sqrtvar = getMu(inputs)

    posTemp = []  # 让学长来加一下

    posTempX = np.tile(posTemp[0], [batchS, 1, 1, 1])
    posTempY = np.tile(posTemp[1], [batchS, 1, 1, 1])

    mask = np.absolute(posTempX - np.reshape(mu_x, [h, w, 1]))  # 对这里[h,w,1]的意义不太确定，不过我觉得1就是指depth，这里不需要batchS参数
    mask = mask + np.absolute(posTempY - np.reshape(mu_y, [h, w, 1]))
    mask = np.maximum(1 - np.multiply(mask, 2), -1)

    # 这些细节需要和学长一起修改：
    # partRate。在大的层面里设置一个与filter_type(partRate)。固定的参数。
    # filter_type应该要被加在our_mnist里面，代表是否需要mask。
    mask[:, :, filter_type != 1, :] = 1
    # print(mask) 看feature map
    return mask


def np_mask(x):
    """
    :param x: an array
    :return: an array
    """
    mask = getMask(x)
    x_new = np.multiply(mask, x)
    return np.maximum(x_new, 0)


# make the date type compatible with tensorflow
np_mask_32 = lambda x: np_mask(x).astype(np.float32)


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
    with ops.name_scope(name, "Mask", [x]) as name:  # where bug is
        z = py_func(np_mask_32,
                    [x],
                    [tf.float32],
                    name=name,
                    grad=our_grad)
        # print(z)  # [<tf.Tensor 'OurRelu:0' shape=<unknown> dtype=float32>]
        z = z[0]  # z is a tensor now
        z.set_shape(x.get_shape())  # We need to give z a shape, in our context, it has the same shape as x
        return z  # Tensor("OurRelu:0", dtype=float32)

        # Test:
        # with tf.Session() as sess:
        #     x = tf.constant([[[1., 2., -3.]], [[7., -8., 9.]]])
        #     x.eval()
        #     z = tf_relu(x)
        #     # z.set_shape(x.get_shape())
        #     gr = tf.gradients(z, [x])[0]
        #     tf.global_variables_initializer().run()
        #     print(x.eval(), z.eval(), gr.eval())