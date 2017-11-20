# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
# from xueting import tf_process_gradient


import tensorflow as tf
import numpy as np
import scipy.io

FLAGS = None

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

def np_mask(x, labels, epoch_):
    """
    :param x: an array
    :return: an array
    """
    printFeatureMap = True
    printMask = False
    printMasked = True
    TensorboardTemplate = False

    if printFeatureMap:
        scipy.io.savemat('featuremap.mat', dict(featuremap=x))

    mask = getMask(x)

    if printMask:
        scipy.io.savemat('mask.mat', dict(mask=mask))
    # print(mask)
    # print(mask.shape)
    if TensorboardTemplate:  # Falied
        t = tf.convert_to_tensor(mask[:, :, :, 0:1], dtype=tf.float32)
        tf.summary.image('template0', t)

    x_new = np.multiply(mask, x)
    x_new = np.maximum(x_new, 0)

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


def gradient2(x, labels, epoch_):
    # MAYBE need to change labels to a numpy array
    batchS, h, w, depth = x.shape
    depthList = np.arange(depth)
    if len(labels.shape) < 2:
        labelNum = 1
    else:
        labelNum = labels.shape[1]
    if labelNum == 1:
        bool_mask = [labels == 1]
        posList = tf.boolean_mask(labels, bool_mask)
        div = [DIV(depthList, posList)]
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

    def setup_logz(mask, theInput, depth, batchS):
        nonlocal alpha
        emp1 = np.zeros((batchS, depth))
        strength = np.mean(np.multiply(theInput, mask),  axis=(1, 2)) + emp1
        # strength now has shape (batchS, depth), value = "spatial" mean of x*mask
        emp2 = np.zeros((1, depth))
        alpha_logZ_pos = np.multiply(np.log(np.mean(
            np.exp(np.divide(np.mean(np.multiply(theInput, mask[::-1, :, :, ::-1]), axis=(1, 2)), alpha)), axis=0)), alpha) \
            + emp2
        # shape is (1, depth)

        alpha_logZ_neg = np.multiply(np.log(np.mean(
            np.exp(np.divide(np.mean(-theInput, axis=(1, 2)), alpha)), axis=0)), alpha) \
            + emp2
        if len(alpha_logZ_pos.shape) != 2:
            raise ValueError('The shape of alpha_logZ_pos is not (1, depth)!')
        # shape is (1, depth)
        # VALUE CHECKING:
        alpha_logZ_pos[np.isinf(alpha_logZ_pos)] = np.amax(alpha_logZ_pos[np.isinf(alpha_logZ_pos) == 0])
        alpha_logZ_neg[np.isinf(alpha_logZ_neg)] = np.amax(alpha_logZ_neg[np.isinf(alpha_logZ_neg) == 0])
        return alpha_logZ_pos, alpha_logZ_neg, strength
    alpha_logZ_pos, alpha_logZ_neg, strength = setup_logz(mask, x, depth, batchS)
    # "strength" is also something that belongs to one input(x)
    # strength has shape (batchS, depth), value = "spatial" mean of x*mask
    # alpha_logZ_pos has shape (1, depth)


    def post_process_gradient(theInput, alpha_logZ_pos, alpha_logZ_neg, div, strength):
        # for lab in range(0, len(div_list)):  (deprecated) in MATLAB div_list is an array with 2 attributes; not a list with 2 elements
        nonlocal depth
        nonlocal imgNum
        nonlocal alpha
        global mag #####################PROBLEM###############################
        grad2 = np.zeros((batchS, h, w, depth))
        # IF div is not a list, len(div) will be 1,
        # IF div is a list (in multi-class case), len(div) will be the length of the list
        for lab in range(len(div)):  # should be 1
            if lab == 1:
                raise ValueError('It is impossible in single class case!')
            if len(div) == 1:
                w_pos = 1
                w_neg = 1
            else:
                raise ValueError('Currently we dont consider multi-class case!')
                # density = end_layer.density  # shape ?
                # w_pos = np.divide(0.5, density[lab])
                # w_neg = np.divide(0.5, 1 - density[lab])

            ## For parts
            # mag ---------- 暂时放一下，不管它，之后调整
            # mag initial = 0.1
            # I temporarily put it in the deepnn function
            # call it a global variable
            # but not sure what this means
            mag = np.divide(np.multiply(np.ones((imgNum, depth)), epoch_), mag)

            dList = div[lab].depthList
            # dList = dList[layer.filters[dList] == 1]    ---------no idea of this.

            if dList.size != 0:  # if dList is not empty
                poslist = div[lab].posList
                neglist = np.setdiff1d(np.arange(batchS), poslist)
                ############################ATTENTION####################################
                # 有一点我觉得很危险，就是poslist和neglist是不是真的如我们所想的是在label中的位置？
                # 但是很奇怪的是MATLAB里的代码中posList好像也不是位置，而是1，2，……396
                # 我觉得这个要确认清楚
                # 目前代码里是把poslist当作一个index在使用的
                # 但是这个很有可能是有问题的。
                #########################################################################
                # theList is a subset of labels (batch)
                # dList is a subset of depth (filters)

                emp3 = np.zeros(poslist.size, dList.size)  # control the shape!
                emp4 = np.zeros((poslist.size, 1, 1, dList.size))
                emp5 = np.zeros((poslist.size, h, w, dList.size))
                emp3_ = np.zeros(neglist.size, dList.size)
                emp4_ = np.zeros((neglist.size, 1, 1, dList.size))
                emp5_ = np.zeros((neglist.size, h, w, dList.size))
                # calculate the gradients for poslist
                if poslist.size != 0:
                    strength = np.multiply(np.exp(np.divide(strength[poslist, dList], alpha)),
                                           strength[poslist, dList] - np.add(alpha_logZ_pos[0, dList], emp3) + alpha)
                    # strength should have the same shape as emp3, (theList.size, dList.size)
                    # we take a subset of the original strength for calculation

                    strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength) == 0])
                    strength[np.isnan(strength)] = 0
                    strength = np.divide(strength, np.multiply(np.mean(strength, 0) + emp3, mag[poslist, dList]) + emp3) + emp4
                    # Now strength has the shape (poslist.size, 1, 1, dList.size)

                    strength[np.isnan(strength)] = 0
                    strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength) == 0])

                    # normalized by mean of strength: problems #####
                    ############################ DOES MASK NEEDED HERE????##############################
                    ################because we have different net structures from MATLAB################
                    updated_value = -np.multiply(np.multiply(mask[poslist, :, :, dList],
                                                             strength + emp5),
                                                 0.00001 * w_pos)
                    grad2[poslist, :, :, dList] += updated_value

                # calculate the gradients for neglist
                if neglist.size != 0:
                    strength = np.mean(theInput[neglist, :, :, dList], axis=(1, 2)) + emp3_
                    strength = np.multiply(np.exp(np.divide(-strength, alpha)),
                                           (-strength - np.add(alpha_logZ_neg[0, dList], emp3_) + alpha))
                    strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength) == 0])
                    strength[np.isnan(strength)] = 0

                    strength = np.divide(strength,
                                         np.multiply(np.mean(strength, 0) + emp3_, mag[neglist, dList] + emp3_)) + emp4_
                    # strength now has the shape (neglist.size, 1, 1, dList.size)
                    strength[np.isnan(strength)] = 0
                    strength[np.isinf(strength)] = np.amax(strength[np.isinf(strength) == 0])

                    updated_value_neg = np.multiply(strength + emp5_, (0.00001 * w_neg))
                    grad2[neglist, :, :, dList] += updated_value_neg

        return grad2

    grad2 = post_process_gradient(x, alpha_logZ_pos, alpha_logZ_neg, div, strength)

    return grad2


def tf_gradient2(x, labels, epoch_, name=None):
    with ops.name_scope(name, "gradient2", [x, labels, epoch_]) as name:
        z = tf.py_func(gradient2,
                           [x, labels, epoch_],
                           [tf.float32],
                           name=name,
                           stateful=False)
        return z[0]
# output的shape不对

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

    n_gr1 = tf_d_mask(x)  # has the same shape as x
    n_gr2 = tf_gradient2(x, labels, epoch_)

    fake_gr1 = labels
    fake_gr2 = epoch_
    return tf.multiply(grad, n_gr1) + n_gr2, fake_gr1, fake_gr2


# input x is a batch of images with only 1 depth
def deepnn(x, labels, epoch_):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    labels: an input tensor with the dimensions(N_examples, ), (in multi-class case, its dimensions: (N_examples, N_classes)
            the labels corresponding to x(which are a batch of images)
            In single-class case, every entry is 1 or -1
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  MASK = True
  zero = True
  global mag
  mag = 0.1  # NO IDEA where to put it
  # labelNum is the classes we want to classify
  # if zero:
  #     labelNum = 1  # single class
  # else:
  #     labelNum = labels.shape[1]

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  if MASK:
    with tf.name_scope('mask'):
        printFetureMap = False
        TensorboardFeatureMap = True
        printMasked = False
        TensorboardMasked = True

        if printFetureMap:
            scipy.io.savemat('premaskk', dict(premaskk=h_conv2))
        if TensorboardFeatureMap:
            for i in range(64):
                tf.summary.image(str.format('premask{}', i), h_conv2[:, :, :, i:i + 1])

        mask = tf_mask(h_conv2, labels, epoch_)  # add labels, epoch_ as inputs

        if printMasked:
            scipy.io.savemat('maskedd', dict(maskedd=mask))
        if TensorboardMasked:
            for i in range(64):
                tf.summary.image(str.format('masked{}', i), mask[:, :, :, i:i + 1])

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(mask)
  else:
      with tf.name_scope('pool2'):
          h_pool2 = max_pool_2x2(h_conv2)


  # Second pooling layer.


  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob', keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  if zero:
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 1])
        b_fc2 = bias_variable([1])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.squeeze(y_conv)
  else:
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def tf_mask(x, labels, epoch_, name=None):  # add "labels" to the input
    with ops.name_scope(name, "Mask", [x, labels, epoch_]) as name:
        z = py_func(np_mask,
                    [x, labels, epoch_],   # add "labels, epoch_" to the input list
                    [tf.float32],
                    name=name,
                    grad=our_grad)
        z = z[0]
        z.set_shape(x.get_shape())
        return z


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  zero = True
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, zero=zero)

  # during Session,
  # feed_dict assgins values to x and y_
  # x are images
  # y are labels

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  if zero:
      y_ = tf.placeholder(tf.float32, [None, ]) # labels

  epoch_ = tf.placeholder(tf.float32)

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x, y_, epoch_)    # provide the deep net graph also with the labels

  # 一开始把问题的解决方法想错了；错误的尝试
  # if zero:
  #     y_conv = tf.argmax(y_conv, axis=1)
  #     # y_conv = tf.cast(y_conv, tf.float32)
  #     Zero = tf.constant(0, dtype=tf.int64)
  #     One = tf.constant(1, dtype=tf.int64)
  #     pos_one = tf.multiply(Zero, y_conv) + One
  #     neg_one = tf.multiply(Zero, y_conv) - One
  #     y_conv = tf.where(tf.equal(y_conv, 0), pos_one, neg_one)
      # y_conv[y_conv != 0].assign(-1.)
      # y_conv[y_conv == 0].assign(1.)
      # tf.nn.sigmoid

  with tf.name_scope('loss'):
    if zero:
        cross_entropy = tf.losses.log_loss(labels=y_,
                                           predictions=y_conv)
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('loss', cross_entropy)


  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  if not zero:
      with tf.name_scope('accuracy'):
          correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
          correct_prediction = tf.cast(correct_prediction, tf.float32)
      accuracy = tf.reduce_mean(correct_prediction)
      tf.summary.scalar('acuuracy', accuracy)


  merged = tf.summary.merge_all()

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location+'/train')
  train_writer.add_graph(tf.get_default_graph())
  # test_writer = tf.summary.FileWriter(graph_location+'/test')


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(501):
      batch = mnist.train.next_batch(50)   # batch[0] are images, batch[1] are labels
      if epoch % 100 == 0:    # every 100 epochs, get a summary
        if zero:
            summary = sess.run(merged, feed_dict={
                x: batch[0], y_: batch[1], epoch_: epoch, keep_prob: 1.0})
        else:
            summary, _ = sess.run([merged, accuracy], feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
        # train_accuracy = accuracy.eval(feed_dict={
        #     x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_writer.add_summary(summary, epoch)
        # summary, acc = sess.run([merged, accuracy], feed_dict={
        #   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        # test_writer.add_summary(summary, i)

        # print('step %d, training accuracy %g' % (i, train_accuracy))

      # the main trainng step
      train_step.run(feed_dict={x: batch[0], y_: batch[1], epoch_: epoch, keep_prob: 0.5})

    # print('test accuracy %g' % acc)
    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def main2(_):
  # Import data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('loss', cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  tf.summary.scalar('acuuracy', accuracy)


  merged = tf.summary.merge_all()

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location+'/train')
  train_writer.add_graph(tf.get_default_graph())
  # test_writer = tf.summary.FileWriter(graph_location+'/test')


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(601):
      batch = mnist.train.next_batch(50)
      if epoch % 100 == 0:
        summary, _ = sess.run([merged, accuracy], feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        # train_accuracy = accuracy.eval(feed_dict={
        #     x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_writer.add_summary(summary, epoch)
        # summary, acc = sess.run([merged, accuracy], feed_dict={
        #   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        # test_writer.add_summary(summary, i)

        # print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


if __name__ == '__main__':
  original_main = True
  if original_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  else:
    tf.app.run(main=main2)
