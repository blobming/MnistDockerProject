import tensorflow as tf
import numpy as np
import cv2


class MnistPredict:

    def __init__(self, filePath):
        self.__filePath = filePath

    @staticmethod
    def imageprepareStorage(file_path):
        picture = cv2.imread(file_path)
        shrink = cv2.resize(picture, (28, 28), interpolation=cv2.INTER_AREA)
        return shrink

    @staticmethod
    def __imageprepare(file_path):
        # file_name = '/Users/xujiaming/MNIST/picture/2.jpg'
        picture = cv2.imread(file_path)
        shrink = cv2.resize(picture, (28, 28), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(shrink, cv2.COLOR_BGR2GRAY)  # 把输入图像灰度化
        if gray[0, 0] < 200:  # 自动修正图片对比度
            gray = np.uint8(np.clip((2.5 * gray + 150), 0, 255))
        # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
        ret, shrink = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        shrink = shrink.reshape(784, )
        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [(255 - x) * 1.0 / 255.0 for x in shrink]
        return np.array(tva)

    @staticmethod
    def __deepnn(x):
        """deepnn builds the graph for a deep net for classifying digits.

      Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.

      Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the digit into one of 10 classes (the
        digits 0-9). keep_prob is a scalar placeholder for the probability of
        dropout.
      """
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            tf.summary.image('input_image', x_image)

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = MnistPredict.__weight_variable([5, 5, 1, 32])
            b_conv1 = MnistPredict.__bias_variable([32])
            h_conv1 = tf.nn.relu(MnistPredict.__conv2d(x_image, W_conv1) + b_conv1)
            tf.summary.histogram('W_conv1', W_conv1)
        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = MnistPredict.__max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = MnistPredict.__weight_variable([5, 5, 32, 64])
            b_conv2 = MnistPredict.__bias_variable([64])
            h_conv2 = tf.nn.relu(MnistPredict.__conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = MnistPredict.__max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = MnistPredict.__weight_variable([7 * 7 * 64, 1024])
            b_fc1 = MnistPredict.__bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = MnistPredict.__weight_variable([1024, 10])
            b_fc2 = MnistPredict.__bias_variable([10])
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv, keep_prob

    @staticmethod
    def __conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def __max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def __weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def __bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def get_predict_result(self):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [None, 784])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])
        y_conv, keep_prob = MnistPredict.__deepnn(x)
        predict = tf.argmax(y_conv, 1)
        print(self.__filePath)
        testImage = MnistPredict.__imageprepare(self.__filePath)

        with tf.Session() as sess:
            #save the model
            saver = tf.train.Saver()
            checkpoint_dir = "ckpt/"
            # return state
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:  # if a model exists
                print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)  # load the model
                predict_result = predict.eval(feed_dict={x: [testImage], keep_prob: 1.0})[0]
            else:
                predict_result = -1
        return predict_result


# only for testing
if __name__ == "__main__":
    a = MnistPredict("/Users/xujiaming/MNIST/picture/2.jpg")
    print(a.get_predict_result())
