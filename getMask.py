import tensorflow as tf
import numpy as np

def getMask(inputs):
    '''
    :param inputs: should be a tensor (last layer's output)
    :return: mask, which is a tensor corresponding to that tensor
    '''
    shape = inputs.get_shape().as_list()  # [batch_size, 7, 7, 64] in mnist example
    batchS, h, w, depth = shape
    def getMu(input = inputs):
        '''
        :param input: tensoor of the layer
        :return: mu_x, mu_y, sqrtvar
        '''
        nonlocal batchS, h, w, depth, shape
        IsUseMax = False
        # 下面是对MATLAb代码的修改
        input = tf.reshape(input, [h * w, depth, batchS])
        if IsUseMax:
            p = np.argmax(input, 0)  # get the index p of input in dimension 1
            p = tf.reshape(p, [batchS, 1, 1, depth])  # p is now an array
            mu_y = np.ceil(np.divide(p, h))    # should return an array the same size of p
            mu_x = p - np.mutiply(mu_y - 1, h)     # the array should be the same size as p
            sqrtvar = []
        else:
            tmp_x = np.array(np.tile(np.transpose(np.arange(1,h)),[batchS, w, depth]))
            tmp_y = np.array(np.reshape(np.tile(np.arange(1,w), [batchS, h, 1, depth]), [batchS, h * w, depth]))
            sumX = np.maximum(np.sum(input, 1), 0.000000001)
            mu_x = np.maximum(round(np.divide(np.sum(np.multiply(tmp_x, input), 1), sumX)) ,1)
            mu_y = np.maximum(round(np.divide(np.sum(np.multiply(tmp_y, input), 1),sumX)), 1)
            sqrtvar = np.sqrt(np.divide((np.sum(np.multiply(np.square(tmp_x - np.tile(mu_x, [1, h * w, 1])) , input), 1) +
                            np.sum(np.multiply(np.square(tmp_y - np.tile(mu_y, [1, h * w, 1])), input), 1)), sumX))
            np.delete(arr=[sumX, tmp_x, tmp_y])

            # It seems the next two lines of codes are not used, I commented them
            # [maxX, ~] = max(x, [], 1)
            # p = np.tile(np.mutilply(mu_x + (mu_y - 1),h), [batchS, 1, 1, depth])

        # (mu_x, mu_y) is the position of mu
        tmp = np.linspace(-1, 1, h)
        mu_x = np.tile(tmp[mu_x], [batchS, 1, 1, depth])
        mu_y = np.tile(tmp[mu_y], [batchS, 1, 1, depth])
        sqrtvar = np.tile(sqrtvar, [batchS, depth])
        return mu_x, mu_y, sqrtvar

    mu_x, mu_y, sqrtvar = getMu(inputs)

    # 然后inputs似乎并没有posTemp, filter, weights这些东西啊

    posTemp = [ , ]  #


    posTempX = np.tile(posTemp[0], [batchS,1, 1, 1])
    posTempY = np.tile(posTemp[1], [batchS, 1, 1, 1])

    mask = np.absolute(posTempX-np.reshape(mu_x, [h, w, 1]))
    mask = mask+np.absolute(posTempY-np.reshape(mu_y, [h, w, 1]))
    #
    mask = np.maximum(1-np.multiply(mask, 2),-1)

    # input.filter 只有等于1的时候才加Loss。
    # partRate。在大的层面里设置一个与filter_type(partRate)。固定的参数。

    mask[:, :, filter_type != 1, :] = 1
    # print(mask) 看feature map
    return mask

# index的顺序



# 一些规律：
# MATLAB --> python
# repmat -> np.tile
# reshape -> np.reshape
# next few functions are all element-wise
# .* -> np.multiply
# ./ -> np.divide
# abs -> np.absolute
# max -> np.maximum
