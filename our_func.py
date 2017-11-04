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
# with tf.Session() as sess:
#     x = tf.constant([[[1., 2., -3.]], [[7., -8., 9.]]])
#     x.eval()
#     z = tf_relu(x)
#     # z.set_shape(x.get_shape())
#     gr = tf.gradients(z, [x])[0]
#     tf.global_variables_initializer().run()
#     print(x.eval(), z.eval(), gr.eval())
