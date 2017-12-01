from tensorflow.python.framework import ops
import matplotlib.pyplot
import tensorflow as tf
import numpy as np
import scipy.io



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
        tmp_x = emp + np.array(np.reshape(np.arange(1, h + 1), newshape=(1, h, 1, 1)))
        tmp_y = emp + np.array(np.reshape(np.arange(1, w + 1), newshape=(1, 1, w, 1)))
        # scipy.io.savemat('tmp_x', dict(tmpx=tmp_x))
        # scipy.io.savemat('tmp_y', dict(tmpy=tmp_y))
        # Now both tmp_x, tmp_y both have shape (batchS, h, w, depth)
        # 我怀疑是tmp_x, tmp_y有问题
        # 它们的作用是提供两组坐标，分别沿着h和w。

        sumX = np.maximum(np.sum(inpu, axis=(1, 2)), 0.000000001)

        # shape of sumX: [batchS, depth]
        tx = np.sum(np.multiply(tmp_x, inpu), (1, 2))
        ty = np.sum(np.multiply(tmp_y, inpu), (1, 2))
        # scipy.io.savemat('tx', dict(tx=tx))
        # scipy.io.savemat('ty', dict(ty=ty))
        # shape of tx, ty: [batchS, depth]

        mu_x = np.maximum(np.round(np.divide(tx, sumX)), 1)
        mu_y = np.maximum(np.round(np.divide(ty, sumX)), 1)


        # 在这一步之前，找mu就已经有问题了。
        # scipy.io.savemat('mu_x1', dict(mu_xint=mu_x))
        # scipy.io.savemat('mu_y1', dict(mu_yint=mu_y))


        # shape of mu_x, mu_y: [batchS, depth]

        # 想要以mu_x, mu_y的各个值为index来取出tmp的元素
        # tmp = np.linspace(-1, 1, h)
        # 肯定可以搞，但暂时没想出怎么用array indexing；不过发现可以直接数学地解决它

        # rescaling
        # Notice that mu_x and mu_y are integers in the range [1, h]
        mu_x = -1 + (mu_x-1) * 2/(h-1)
        mu_y = -1 + (mu_y-1) * 2/(w-1)
        # values are between [-1, 1]
        # shape of mu_x, mu_y: [batchS, depth]
        mu_x = mu_x.reshape((batchS, 1, 1, depth))
        mu_y = mu_y.reshape((batchS, 1, 1, depth))
        # scipy.io.savemat('mu_x2', dict(mu_xfloat=mu_x))
        # scipy.io.savemat('mu_y2', dict(mu_yfloat=mu_y))

        return mu_x, mu_y


    # mu_x, mu_y should be batch-wise, depth-wise
    # different batches are different images, different depths are different filters

    mu_x, mu_y = getMu(inputs)
    # mu_x, mu_y 's shapes are both (batchS, 1, 1, depth)

    # 设一个index作为全局变量，每一层存一个对应的posTemp
    lx = np.linspace(-1, 1, h)
    lx = np.reshape(lx, newshape=(1, h, 1, 1))
    ly = np.linspace(-1, 1, w)
    ly = np.reshape(ly, newshape=(1, 1, w, 1))
    empTemp = np.zeros((batchS, h, w, depth))
    posTempX = empTemp + lx
    posTempY = empTemp + ly

    mu_x = empTemp + mu_x
    mu_y = empTemp + mu_y


    # mu_x, mu_y 's shapes are both [batchS, h, w, depth]
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
    # mask shape: [batchS, h, w, depth]

def np_mask(x, labels, epoch_, mag):
    """
    :param x: an array
    :return: an array
    """
    printFeatureMap = True
    printMask = False
    printMasked = True
    TensorboardTemplate = False

    if printFeatureMap and epoch_ % 100 == 0:
        for i in range(20):
            for j in range(5):
                # scipy.io.savemat('featuremap{} for image{}'.format(10*j, 2*i), dict(featuremap=x[2*i, :, :, 10*j]))
                matplotlib.pyplot.imsave('featuremap{} for image{}.jpg'.format(10*j, 2*i), x[2*i, :, :, 10*j])
        scipy.io.savemat('featuremap.mat', dict(featuremap=x))

    mask = getMask(x)


    # print(mask)
    # print(mask.shape)
    if TensorboardTemplate:  # Falied
        t = tf.convert_to_tensor(mask[:, :, :, 0:1], dtype=tf.float32)
        tf.summary.image('template0', t)

    x_new = np.multiply(mask, x)
    x_new = np.maximum(x_new, 0)

    if printMasked and epoch_ % 100 == 0:
        for i in range(20):
            for j in range(5):
                # scipy.io.savemat('maskedmap{} for image{}'.format(10*j, 2*i), dict(maskedmap=x_new[2*i, :, :, 10*j]))
                # scipy.misc.toimage(x_new[2 * i, :, :, 10 * j]).save('maskedmap{} for image{}.jpg'.format(10*j, 2*i))
                matplotlib.pyplot.imsave('maskedmap{} for image{}.jpg'.format(10*j, 2*i), x_new[2*i, :, :, 10*j])


    if printMasked:
        scipy.io.savemat('masked.mat', dict(masked=x_new))

    return x_new.astype(np.float32)

np_mask_test = False
if np_mask_test:
    test = np.arange(1, 61)
    test = np.reshape(test, [3, 2, 2, 5])
    np_mask(test)

def d_mask(x):
    """
    CURRENTLY,
    :param x: a number
    :return:  a number
    BUT as long as we can define this function with array input and array output, i.e. a numpy function,
    We don't need to vectorize it later.
    """
    OLD_return = False
    mask = getMask(x)

    if OLD_return:
        return np.multiply(mask, np.maximum(mask, 0))
    else:
        return np.maximum(mask, 0)

d_mask_32 = lambda x: d_mask(x).astype(np.float32)  # make data type compatible

# transform the numpy function into a Tensorflow function
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
        # currently have no idea of how to deal with
        # MULTI-CLASS case for DIV object
        # because at that time we want it to have larger lengths
        # maybe we can define more properties


def gradient2(x, labels, epoch_, mag):
    batchS, h, w, depth = x.shape
    depthList = np.arange(depth)
    epoch_ += 1  # we don't want epoch_ = 0
    if len(labels.shape) < 2:
        labelNum = 1
    else:
        labelNum = labels.shape[1]
    if labelNum == 1:
        labels = np.squeeze(labels)
        bool_mask = [labels == 1]
        # print(bool_mask)  # shape (50,)
        # labels = tf.convert_to_tensor(labels, tf.float32)
        # print(labels)
        # labels shape (1, 50)
        label_index = np.arange(labels.size)
        posList = label_index[bool_mask]
        # print(2)
        # This way we can only get a list full of 1
        # I think we should change the "labels" to np.arange(labels.size)

        div = [DIV(depthList, posList)]
        # print('Writing div')
        # f1 = open('div.txt', 'w')
        # f1.write(str(div))
        # f1.close()
        # print(3)

        # different data structue to store DIV from MATLAB
        # since depthList and posList have different lengths,
        # we cannot put them into one array
    else:
        raise ValueError('labelNum is not 1!!!')
    # can make div a list
    # layer, end_layer = initialization()  # 这个function要改在外面declare
    # div_list = setup_div(layer, end_layer)
    imgNum = batchS
    alpha = 0.5
    mask = getMask(x)
    # print('hey')
###########################################################################################
    ##################################### 11.19 ######################################
    ###########################有很多data type和shape的细节问题##########################

    def setup_logz(mask, theInput, depth, batchS):
        nonlocal alpha
        # f = open("setup_logz1.txt", "w")
        emp1 = np.zeros((batchS, depth))
        # print('setup logz')
        strength = np.mean(np.multiply(theInput, mask),  axis=(1, 2)) + emp1
        # print('Writing strength to the file...')
        # f.write(str(strength) + "\n")
        # print('calculate strength')
        # strength now has shape (batchS, depth), value = "spatial" mean of x*mask
        emp2 = np.zeros((1, depth))
        alpha_logZ_pos = np.multiply(np.log(np.mean(
            np.exp(np.divide(np.mean(np.multiply(theInput, mask[::-1, :, :, ::-1]), axis=(1, 2)), alpha)), axis=0)), alpha) \
            + emp2
        # print('Writing alpha_logZ_pos to the file...')
        # f.write(str(alpha_logZ_pos) + "\n")
        # shape is (1, depth)

        alpha_logZ_neg = np.multiply(np.log(np.mean(
            np.exp(np.divide(np.mean(-theInput, axis=(1, 2)), alpha)), axis=0)), alpha) \
            + emp2
        # print('Writing alpha_logZ_neg to the file...')
        # f.write(str(alpha_logZ_neg) + "\n")

        if len(alpha_logZ_pos.shape) != 2:
            raise ValueError('The shape of alpha_logZ_pos is not (1, depth)!')
        # shape is (1, depth)

        # VALUE CHECKING:########################################
        # alpha_logZ_pos[np.isinf(alpha_logZ_pos)] = np.amax(alpha_logZ_pos[np.isinf(alpha_logZ_pos) == 0])
        # alpha_logZ_neg[np.isinf(alpha_logZ_neg)] = np.amax(alpha_logZ_neg[np.isinf(alpha_logZ_neg) == 0])
        #########################################################
        # f.close()
        return alpha_logZ_pos, alpha_logZ_neg, strength

    # print('run this')
    # print(mask, x, depth, batchS)
    # array, array, 64, 50
    alpha_logZ_pos, alpha_logZ_neg, strength = setup_logz(mask, x, depth, batchS)
    # print('finish setup_logz')
    # "strength" is also something that belongs to one input(x)
    # strength has shape (batchS, depth), value = "spatial" mean of x*mask
    # alpha_logZ_pos has shape (1, depth)

    # f = open('post_process_gradient.txt', 'w')
    # f.write('Initital:\n'+str(alpha_logZ_pos)+'\n' +
    #         str(alpha_logZ_neg)+'\n' +
    #         str(strength)+'\n')

    def post_process_gradient(theInput, alpha_logZ_pos, alpha_logZ_neg, div, strength):
        nonlocal depth
        nonlocal imgNum
        nonlocal alpha
        nonlocal h
        nonlocal w
        nonlocal alpha
        nonlocal mag
        # print('Start pose_process_gradient')
        grad2 = np.zeros((batchS, h, w, depth))
        # print('Writing initial grad2...')
        # f.write(np.array_str(grad2)+'\n')
        # np.savetxt('grad2', grad2)

        # IF div is not a list, len(div) will be 1,
        # IF div is a list (in multi-class case), len(div) will be the length of the list
        # print('Setting up w_pos and w_neg')
        for lab in range(len(div)):  # should be 1
            if lab == 1:
                raise ValueError('It is impossible in single class case!')
            if len(div) == 1:
                w_pos = 1
                w_neg = 1
            else:
                raise ValueError('Currently we do not consider multi-class case!')
                # density = end_layer.density  # shape ?
                # w_pos = np.divide(0.5, density[lab])
                # w_neg = np.divide(0.5, 1 - density[lab])
            ## For parts
            # mag initial = 0.1
            # I temporarily put it in the deepnn function
            # call it a global variable
            # but not sure what this means
            mag = np.divide(np.multiply(np.ones((imgNum, depth)), epoch_), mag)
            # print('Writing mag...')
            # f.write(np.array_str(mag)+'\n')
            # np.savetxt('mag', mag)

            dList = div[lab].depthList
            # print('check Dlist again')
            # f.write(np.array_str(dList) + '\n')
            # dList = dList[layer.filters[dList] == 1]    ---------no idea of this.

            if dList.size != 0:  # if dList is not empty
                poslist = div[lab].posList
                neglist = np.setdiff1d(np.arange(batchS), poslist)
                # print('Writing poslist and neglist...')
                # f.write(np.array_str(poslist)+'\n' +
                #         np.array_str(neglist)+'\n')
                ############################ATTENTION####################################
                # 有一点我觉得很危险，就是poslist和neglist是不是真的如我们所想的是在label中的位置？
                # 但是很奇怪的是MATLAB里的代码中posList好像也不是位置，而是1，2，……396
                # 我觉得这个要确认清楚
                # 目前代码里是把poslist当作一个index在使用的
                # 但是这个很有可能是有问题的。
                #########################################################################
                # theList is a subset of labels (batch)
                # dList is a subset of depth (filters)
                # print(poslist.size)  # 5
                # print(neglist.size)  # 45
                # print(dList.size)    # 64
                # print(h, w)          # 14, 14
                emp3 = np.zeros((poslist.size, dList.size))  # control the shape!
                emp4 = np.zeros((poslist.size, 1, 1, dList.size))
                emp5 = np.zeros((poslist.size, h, w, dList.size))
                emp3_ = np.zeros((neglist.size, dList.size))
                emp4_ = np.zeros((neglist.size, 1, 1, dList.size))
                emp5_ = np.zeros((neglist.size, h, w, dList.size))
                # print(emp3, emp3_)
                # print('Calculate the gradients for poslist')
                ############################################ 11.20 before dinner ##############################
                ############################################### mark progress #################################

                if poslist.size != 0:
                    # print('run into first if conditional')
                    # print(alpha_logZ_pos.shape)  # (1, 64)
                    # print(strength.shape)     # (50, 64)
                    # print(poslist, dList)
                    # print(poslist.shape, dList.shape)  # (8,) (64,)
                    # print(strength[poslist, dList])  # Problem!
                    # rows = strength[poslist][:, dList]
                    # print(rows[:, dList])
                    # what we want: strength to be a array with shape (poslist.size, dList.size)


                    # add1 = np.exp(np.divide(strength[poslist][:, dList], alpha))
                    # print(add1.shape)
                    # add2 = strength[poslist][:, dList] - np.add(alpha_logZ_pos[0, dList], emp3) + alpha
                    # print(add2)
                    strength = np.multiply(np.exp(np.divide(strength[poslist][:, dList], alpha)),
                                           strength[poslist][:, dList] - np.add(alpha_logZ_pos[0, dList], emp3) + alpha)
                    # print('strength1')
                    # print(strength.shape)  # (5, 64)
                    # strength should have the same shape as emp3, (theList.size, dList.size)
                    # we take a subset of the original strength for calculation

                    ##################################################################
                    # strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength) == 0])
                    # strength[np.isnan(strength)] = 0
                    #################################################################
                    # print(np.add(np.mean(strength, 0)+emp3).shape)

                    strength = np.reshape(np.divide(strength, np.multiply(np.mean(strength, 0)+emp3, mag[poslist][:, dList]) + emp3),
                                          (poslist.size, 1, 1, depthList.size))

                    # Now strength has the shape (poslist.size, 1, 1, dList.size)
                    # print('strength2')
                    # print(strength.shape)

                    ######################################################
                    # strength[np.isnan(strength)] = 0
                    # strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength) == 0])
                    ########################################################

                    # normalized by mean of strength: problems #####
                    ############################## IS MASK NEEDED HERE????##############################
                    ################because we have different net structures from MATLAB################

                    # a = strength + emp5
                    # print(a.shape)
                    updated_value = -np.multiply(np.multiply(mask[poslist][:, :, :, dList],
                                                             strength + emp5),
                                                 0.00001 * w_pos)
                    # print(updated_value.shape)
                    grad2[poslist][:, :, :, dList] += updated_value

                # print('Calculate the gradients for neglist')
                if neglist.size != 0:
                    strength = np.mean(theInput[neglist][:, :, :, dList], axis=(1, 2)) + emp3_
                    # print('strength0_')
                    # print(strength.shape)
                    strength = np.multiply(np.exp(np.divide(-strength, alpha)),
                                           (-strength - np.add(alpha_logZ_neg[0, dList], emp3_) + alpha))
                    # print('strength1_')
                    # print(strength.shape)
                    #####################################################
                    # strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength) == 0])
                    # strength[np.isnan(strength)] = 0
                    #####################################################

                    strength = np.reshape(np.divide(strength,
                                         np.multiply(np.mean(strength, 0) + emp3_, mag[neglist][:, dList] + emp3_)),
                                          (neglist.size, 1, 1, depthList.size))
                    # print('strength2_')
                    # print(strength.shape)
                    # strength now has the shape (neglist.size, 1, 1, dList.size)
                    ###################################################################
                    # strength[np.isnan(strength)] = 0
                    # strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength) == 0])
                    ###################################################################

                    updated_value_neg = np.multiply(strength + emp5_, (0.00001 * w_neg))
                    # print(updated_value_neg.shape)
                    grad2[neglist][:, :, :, dList] += updated_value_neg
                    # print(grad2.shape)
        return grad2

    # f.close()

    grad2 = post_process_gradient(x, alpha_logZ_pos, alpha_logZ_neg, div, strength)
    # print('Finally we reach here!')
    # print(grad2.shape)
    grad2 = grad2.astype(np.float32)
    # print(grad2.dtype)
    return grad2


def tf_gradient2(x, labels, epoch_, mag, name=None):
    with ops.name_scope(name, "gradient2", [x, labels, epoch_, mag]) as name:
        z = tf.py_func(gradient2,
                           inp=[x, tf.convert_to_tensor(labels, tf.float32), epoch_, mag],
                           Tout=[tf.float32],
                           name=name,
                           stateful=False)
        return z[0]

# tf.py_func acts on lists of tensors (and returns a list of tensors),
# that is why we have [x] (and return z[0]).

# grad = cus_op.inputs[0]
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
    labels = cus_op.inputs[1]  # when calculate n_gr2, we will need the labels
    epoch_ = cus_op.inputs[2]  ########################         and the epoch
    mag = cus_op.inputs[3]

    n_gr1 = tf_d_mask(x)  # has the same shape as x
    n_gr2 = tf_gradient2(x, labels, epoch_, mag)

    fake_gr1 = labels
    fake_gr2 = epoch_
    fake_gr3 = mag

    return tf.multiply(grad, n_gr1) + n_gr2, fake_gr1, fake_gr2, fake_gr3

def tf_mask(x, labels, epoch_, mag, name=None):  # add "labels" to the input
    with ops.name_scope(name, "Mask", [x, labels, epoch_, mag]) as name:
        z = py_func(np_mask,
                    [x, tf.convert_to_tensor(labels, tf.float32), epoch_, mag],   # add "labels, epoch_" to the input list
                    [tf.float32],
                    name=name,
                    grad=our_grad)
        z = z[0]
        z.set_shape(x.get_shape())
        return z