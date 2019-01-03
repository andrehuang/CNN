from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np


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
    :return: mask, which is a tensor corresponding to that tensor (and has the same shape as the input tensor)
    '''
    shape = inputs.shape
    batchS, h, w, depth = shape
    Lambda = 2

    def getMu(inpu=inputs):
        '''
        :param input: tensor , the layer
        :return: mu_x, mu_y; (mu_x, mu_y) is the location of the mu;
                the shapes of mu_x, mu_y are both [batchS, 1, 1, depth]
        '''

        tmp_x = np.tile(np.array(np.reshape(np.arange(1, h + 1), newshape=(1, h, 1, 1))), (batchS, 1, w, depth))
        tmp_y = np.tile(np.array(np.reshape(np.arange(1, w + 1), newshape=(1, 1, w, 1))), (batchS, h, 1, depth))

        sumX = np.maximum(np.sum(inpu, axis=(1, 2)), 0.000000001)

        # shape of sumX: [batchS, depth]
        tx = np.sum(np.multiply(tmp_x, inpu), (1, 2))
        ty = np.sum(np.multiply(tmp_y, inpu), (1, 2))
        # shape of tx, ty: [batchS, depth]

        mu_x = np.maximum(np.round(np.divide(tx, sumX)), 1)
        mu_y = np.maximum(np.round(np.divide(ty, sumX)), 1)


        # shape of mu_x, mu_y: [batchS, depth]
        # rescaling
        # Notice that mu_x and mu_y are integers in the range [1, h]
        mu_x = -1 + (mu_x-1) * 2/(h-1)
        mu_y = -1 + (mu_y-1) * 2/(w-1)
        # values are between [-1, 1]
        # shape of mu_x, mu_y: [batchS, depth]
        mu_x = mu_x.reshape((batchS, 1, 1, depth))
        mu_y = mu_y.reshape((batchS, 1, 1, depth))

        return mu_x, mu_y


    mu_x, mu_y = getMu(inputs)
    # mu_x, mu_y 's shapes are both (batchS, 1, 1, depth)

    lx = np.reshape(np.linspace(-1, 1, h), newshape=(1, h, 1, 1))
    ly = np.reshape(np.linspace(-1, 1, w), newshape=(1, 1, w, 1))
    posTempX = np.tile(lx, (batchS, 1, w, depth))
    posTempY = np.tile(ly, (batchS, h, 1, depth))

    mu_x = np.tile(mu_x, (1, h, w, 1))
    mu_y = np.tile(mu_y, (1, h, w, 1))
    # mu_x, mu_y 's shapes are both [batchS, h, w, depth]

    mask = np.absolute(posTempX - mu_x)
    mask = np.add(mask, np.absolute(posTempY - mu_y))
    mask = np.maximum(1 - np.multiply(mask, Lambda), -1)

    return mask
    # mask shape: [batchS, h, w, depth]


def np_mask(x, labels, epoch_, mag):
    """
    :param x: an array (feature map)
    :return: an array (masked feature map)
    """
    mask = getMask(x)
    x_new = np.multiply(mask, x)
    x_new = np.maximum(x_new, 0)
    return x_new.astype(np.float32)


def d_mask(x):
    """
    CURRENTLY,
    :param x: a number
    :return:  a number
    BUT as long as we can define this function with array input and array output, i.e. a numpy function,
    We don't need to vectorize it later.
    """
    mask = getMask(x)
    return np.maximum(mask, 0)


d_mask_32 = lambda x: d_mask(x).astype(np.float32)  # make data type compatible

# transform the numpy function into a TensorFlow function
def tf_d_mask(x, name=None):
    """
    :param x: a list of tensors (but we can just see it as a tensor)
    :param name: 
    :return: a tensor
    """
    with ops.name_scope(name, "d_mask", [x]) as name:
        z = tf.py_func(d_mask_32,
                       [x],
                       tf.float32,
                       name=name,
                       stateful=False)
        return z[0]


class DIV:  # initialize Div struct by depthList and posList
    def __init__(self, depthList, posList):
        self.depthList = depthList
        self.posList = posList
        self.length = 1


def gradient2(x, labels, epoch_, mag):
    batchS, h, w, depth = x.shape
    depthList = np.arange(depth)
    epoch_copy = epoch_ + 1
    if len(labels.shape) >= 2:
        raise ValueError('labelNum is not 1!!!')

    labels = np.squeeze(labels)
    bool_mask = [labels == 1]
    label_index = np.arange(labels.size)
    posList = label_index[bool_mask]

    div = [DIV(depthList, posList)]

    imgNum = batchS
    alpha = 0.5
    mask = getMask(x)

    def setup_logz(mask, theInput, depth, batchS):
        MaxValue = 1000000
        strength =np.reshape(np.mean(np.multiply(theInput, mask),  axis=(1, 2)), (batchS, depth))
        alpha_logZ_pos = np.reshape(np.multiply(np.log(np.mean(
            np.exp(np.divide(np.mean(np.multiply(theInput, mask[::-1, :, :, ::-1]), axis=(1, 2)), alpha)), axis=0)), alpha), (1, depth))

        alpha_logZ_neg = np.reshape(np.multiply(np.log(np.mean(
            np.exp(np.divide(np.mean(-theInput, axis=(1, 2)), alpha)), axis=0)), alpha), (1, depth))

        alpha_logZ_pos = np.minimum(alpha_logZ_pos, MaxValue)
        alpha_logZ_neg = np.minimum(alpha_logZ_neg, MaxValue)

        if len(alpha_logZ_pos.shape) != 2:
            raise ValueError('The shape of alpha_logZ_pos is not (1, depth)!')
        return alpha_logZ_pos, alpha_logZ_neg, strength

    alpha_logZ_pos, alpha_logZ_neg, strength = setup_logz(mask, x, depth, batchS)
    # alpha_logZ_pos, alpha_logZ_neg : (1, depth)
    # strength: (batchS, depth)

    def post_process_gradient(theInput, alpha_logZ_pos, alpha_logZ_neg, div, strength, mag):
        grad2 = np.zeros((batchS, h, w, depth))
        MaxValue = 1000000
        for lab in range(len(div)):  # should be 1
            if lab == 1:
                raise ValueError('It is impossible in single class case!')

            if len(div) == 1:
                w_pos = 1
                w_neg = 1
            else:
                raise ValueError('Currently we do not consider multi-class case!')

            mag = np.divide(np.multiply(np.ones((imgNum, depth)), epoch_copy), mag)

            dList = div[lab].depthList

            if dList.size != 0:
                poslist = div[lab].posList
                neglist = np.setdiff1d(np.arange(batchS), poslist)
                length_pos = len(poslist)
                length_neg = len(neglist)
                length_dlist = len(dList)

                if poslist.size != 0:
                    strength = np.multiply(np.exp(np.divide(strength[poslist, :][:, dList], alpha)),
                                           strength[poslist, :][:, dList] - np.tile(alpha_logZ_pos[0, dList], (length_pos, 1)) + alpha)
                    strength = np.minimum(strength, MaxValue)

                    strength = np.reshape(
                        np.divide(strength,
                                  np.reshape(np.multiply(np.tile(np.mean(strength, 0), (length_pos, 1)),
                                                         mag[poslist, :][:, dList]), (length_pos, length_dlist))),
                        (length_pos, 1, 1, length_dlist))
                    strength = np.minimum(strength, MaxValue)

                    updated_value = -np.multiply(np.multiply(mask[poslist][:, :, :, dList],
                                                             np.tile(strength, (1, h, w, 1))),
                                                 0.00001 * w_pos)
                    grad2[poslist][:, :, :, dList] += updated_value

                if neglist.size != 0:
                    strength = np.reshape(np.mean(theInput[neglist][:, :, :, dList], axis=(1, 2)), (length_neg, length_dlist))

                    strength = np.multiply(np.exp(np.divide(-strength, alpha)),
                                           (-strength - np.tile(alpha_logZ_neg[0, dList], (length_neg, 1)) + alpha))
                    strength = np.minimum(strength, MaxValue)

                    strength = np.reshape(
                        np.divide(strength,
                                  np.reshape(np.multiply(np.tile(np.mean(strength, 0), (length_neg, 1)),
                                                         mag[neglist, :][:, dList]), (length_neg, length_dlist))),
                        (length_neg, 1, 1, length_dlist))
                    strength = np.minimum(strength, MaxValue)

                    updated_value_neg = np.multiply(np.tile(strength, (1, h, w, 1)), (0.00001 * w_neg))
                    grad2[neglist][:, :, :, dList] += updated_value_neg
        return grad2

    gr2 = post_process_gradient(x, alpha_logZ_pos, alpha_logZ_neg, div, strength, mag)
    gr2 = gr2.astype(np.float32)
    return gr2


def tf_gradient2(x, labels, epoch_, mag, name=None):
    with ops.name_scope(name, "gradient2", [x, labels, epoch_, mag]) as name:
        z = tf.py_func(gradient2,
                           inp=[x, tf.convert_to_tensor(labels, tf.float32), epoch_, mag],
                           Tout=[tf.float32],
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
    labels = cus_op.inputs[1]
    epoch_ = cus_op.inputs[2]
    mag = cus_op.inputs[3]

    n_gr1 = tf_d_mask(x)

    n_gr2 = tf_gradient2(x, labels, epoch_, mag)

    fake_gr1 = labels
    fake_gr2 = epoch_
    fake_gr3 = mag

    return tf.multiply(grad, n_gr1) + n_gr2, fake_gr1, fake_gr2, fake_gr3


def tf_mask(x, labels, epoch_, mag, name=None):
    with ops.name_scope(name, "Mask", [x, labels, epoch_, mag]) as name:
        z = py_func(np_mask,
                    [x, tf.convert_to_tensor(labels, tf.float32), epoch_, mag],
                    [tf.float32],
                    name=name,
                    grad=our_grad)

        z = z[0]
        z.set_shape(x.get_shape())
        return z