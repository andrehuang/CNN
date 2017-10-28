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

def np_relu(x):
    """
    :param x: a Tensor object
    :return: 
    """
    y = np.maximum(x, 0)
    return y

# np_relu = np.vectorize(our_relu)
np_relu_32 = lambda x: np_relu(x).astype(np.float32)

# Next,
# define the gradient function of our operation for each input
def d_relu(x):
    if x > 0:
        return 1
    else:
        return 0
np_d_relu = np.vectorize(d_relu)
# vectorizing: making it into a numpy function

np_d_relu_32 = lambda x: np_d_relu(x).astype(np.float32)
# One thing to be careful of at this point is that numpy used float64
# but tensorflow uses float32 so you need to convert your function
# to use float32 before you can convert it to a tensorflow function
# otherwise tensorflow will complain.


# Notice that we need to return tensorflow functions of the input
# transform the numpy function into a Tensorflow function
def tf_d_relu(x, name=None):
    with ops.name_scope(name, "d_relu", [x]) as name:   # where bug is
        z = tf.py_func(np_d_relu_32,
                    [x],
                    tf.float32,
                    name=name,
                    stateful=False)
        return z[0]
# tf.py_func acts on lists of tensors (and returns a list of tensors),
# that is why we have [x] (and return y[0]).


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
# NEEDS POLISHED! ESPECIALLY THE SHAPE NEEDS TO MATCH!

# our hack
def py_func(func, inp, tout, stateful=True, name=None, grad=None):
    """
    :param func: 
    :param inp: 
    :param tout: 
    :param stateful: tells the tensorflow whether the function always 
        gives the same output for the same input, 
        in which case tensorflow can simplify the tensorflow graph.
        
        BUT We don't really care about this detail right now!
    :param name: 
    :param grad: 
    :return: 
    """
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 100000))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, tout, stateful=stateful, name=name)


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
# DON'T UNDERSTAND:
# tf.py_func acts on lists of tensors (and returns a list of tensors),
# that is why we have [x,y] (and return z[0])

# Test:
# with tf.Session() as sess:
#     x = tf.constant([[[1., 2., -3.]], [[7., -8., 9.]]])
#     x.eval()
#     z = tf_relu(x)
#     # z.set_shape(x.get_shape())
#     gr = tf.gradients(z, [x])[0]
#     tf.global_variables_initializer().run()
#     print(x.eval(), z.eval(), gr.eval())
    #print(x.eval(), z.eval())
# After debugging the line100, we can try substitute the relu function in cnn_mnist with this customized tf_relu
# to see whether this method works or not.
