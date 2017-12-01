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
from mask_operation import tf_mask

import tensorflow as tf
import scipy.io
import scipy.misc



FLAGS = None

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
        TensorboardFeatureMap = False
        printMasked = False
        TensorboardMasked = False

        if printFetureMap:
            scipy.io.savemat('premaskk', dict(premaskk=h_conv2))
        if TensorboardFeatureMap:
            for i in range(64):
                tf.summary.image(str.format('premask{}', i), h_conv2[:, :, :, i:i + 1])

        mask = tf_mask(h_conv2, labels, epoch_, mag)  # add labels, epoch_ as inputs

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

  # if epoch_ == 400:
  #     np.savetxt('final_y_conv.txt', y_conv)


  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


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
  trial_loss = True
  # Import data
  mnist = input_data.our_read_data_sets(FLAGS.data_dir, one_hot=True, zero=zero)

  # during Session,
  # feed_dict assgins values to x and y_
  # x are images
  # y are labels

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  if zero:
      y_ = tf.placeholder(tf.float32, [None, ])  # labels

  epoch_ = tf.placeholder(tf.float32, name='Epoch')
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
        if trial_loss:
            cross_entropy = tf.log(1 + tf.exp(-tf.multiply(y_, y_conv)))
            # cross_entropy = tf.losses.mean_squared_error(labels=y_,
            #                                              predictions=y_conv)
    else:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
  # print(cross_entropy)
  cross_entropy = tf.reduce_mean(cross_entropy)
  # print(cross_entropy)
  tf.summary.scalar('loss', cross_entropy)
  # Finally I use MSE as loss function temporarily, avoiding the tround caused by -1.



  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        if zero:
            correct_prediction = tf.equal(tf.sign(y_conv), tf.sign(y_))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        else:
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
    for epoch in range(401):
      batch = mnist.train.next_batch(50)   # batch[0] are images, batch[1] are labels
      if epoch % 100 == 0:    # every 100 epochs, get a summary
        if zero:
            # if trial_loss:
            #     cross_entropy1 = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_,
            #                                predictions=y_conv))
            # else:
            #     cross_entropy1 = tf.reduce_mean(tf.losses.log_loss(labels=y_,
            #                                predictions=y_conv))
            # Loss can not be calculated bacause y_conv has -1 which can not be taken log.
            [summary, loss_val, acc] = sess.run([merged, cross_entropy, accuracy], feed_dict={
                x: batch[0], y_: batch[1], epoch_: epoch, keep_prob: 1.0})
            print('epoch {}, loss {}'.format(epoch, loss_val))
            print('epoch {}, training accuracy {}'.format(epoch, acc))
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


      # 这个还不一定奏效！
      # y_conv_final = y_conv.eval()
      # y_conv_final = np.asarray(y_conv_final)
      # np.savetxt('y_conv_final.txt', y_conv_final)

      if epoch % 100 == 0:
          images = batch[0]
          for i in range(20):
            image = images[2*i, :]
            image = tf.reshape(image, (28, 28, 1))
            image = tf.multiply(image, 255)
            image = tf.cast(image, tf.uint8)
            enc = tf.image.encode_jpeg(image)
            fname = tf.constant('image{}.jpg'.format(2*i))
            fwrite = tf.write_file(fname, enc)
            sess.run(fwrite)


    # print('test accuracy %g' % acc)
    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# def main2(_):
#   # Import data
#   mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#   train_data = mnist.train.images  # Returns np.array
#   train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#   eval_data = mnist.test.images  # Returns np.array
#   eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#   # Create the model
#   x = tf.placeholder(tf.float32, [None, 784])
#
#   # Define loss and optimizer
#   y_ = tf.placeholder(tf.float32, [None, 10])
#
#   # Build the graph for the deep net
#   y_conv, keep_prob = deepnn(x)
#
#   with tf.name_scope('loss'):
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
#                                                             logits=y_conv)
#   cross_entropy = tf.reduce_mean(cross_entropy)
#   tf.summary.scalar('loss', cross_entropy)
#
#   with tf.name_scope('adam_optimizer'):
#     train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
#   with tf.name_scope('accuracy'):
#     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#     correct_prediction = tf.cast(correct_prediction, tf.float32)
#   accuracy = tf.reduce_mean(correct_prediction)
#   tf.summary.scalar('acuuracy', accuracy)
#
#
#   merged = tf.summary.merge_all()
#
#   graph_location = tempfile.mkdtemp()
#   print('Saving graph to: %s' % graph_location)
#   train_writer = tf.summary.FileWriter(graph_location+'/train')
#   train_writer.add_graph(tf.get_default_graph())
#   # test_writer = tf.summary.FileWriter(graph_location+'/test')
#
#
#   with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(601):
#       batch = mnist.train.next_batch(50)
#       if epoch % 100 == 0:
#         summary, _ = sess.run([merged, accuracy], feed_dict={
#             x: batch[0], y_: batch[1], keep_prob: 1.0})
#         # train_accuracy = accuracy.eval(feed_dict={
#         #     x: batch[0], y_: batch[1], keep_prob: 1.0})
#         train_writer.add_summary(summary, epoch)
#         # summary, acc = sess.run([merged, accuracy], feed_dict={
#         #   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
#         # test_writer.add_summary(summary, i)
#
#         # print('step %d, training accuracy %g' % (i, train_accuracy))
#       train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


if __name__ == '__main__':
  original_main = True
  if original_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# 学习一个！
# The reason you are getting NaN's is most likely that somewhere
# in your cost function or softmax you are trying to take a log of zero,
# which is not a number.
