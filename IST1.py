'''
    Image Style Transfer Using Convolutional Neural Networks Using vgg16
'''

import tensorflow as tf
from tensorflow.nn import conv2d, bias_add, relu, max_pool, l2_loss
from scipy.misc import imread, imresize, imsave
import numpy as np
import sys


def load_vgg16_weights(vgg16_path):
    return np.load(vgg16_path).item()


def load_img(content_file, style_file):
    content_img = imread(content_file)
    style_img = imread(style_file)
    content_wh = content_img.shape[:2]
    return content_wh, content_img, style_img


def vgg16_preprocess(img, resize_wh):
    img = imresize(img, (resize_wh, resize_wh)).astype('float32')
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    return img[:, :, ::-1]


def vgg16_depreprocess(img, content_wh):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    return imresize(img[:, :, ::-1].clip(0, 255).astype('uint8'), content_wh)


class ImageStyleTransfer():
    def __init__(self, vgg16_weights, content_img_value, style_img_value, lr, content_weight, style_weight,
                 graph_write):
        self.vgg16_weights = vgg16_weights
        self.content_img_value = content_img_value
        self.style_img_value = style_img_value
        self.lr = lr
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.graph_write = graph_write

        self.build_model()

    def build_model(self):
        with tf.name_scope('Input'):
            content_img = tf.constant(self.content_img_value, name='content_image')
            style_img = tf.constant(self.style_img_value, name='style_image')
            self.magical_img = tf.Variable(initial_value=content_img, name='magical_image')
            input_tensor = tf.concat([tf.expand_dims(content_img, axis=0), tf.expand_dims(style_img, axis=0),
                                      tf.expand_dims(self.magical_img, axis=0)], axis=0)

        with tf.name_scope('conv1'):
            conv1_1 = relu(bias_add(
                conv2d(input_tensor, tf.constant(self.vgg16_weights['block1_conv1'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block1_conv1'][1]), name='conv1_1')
            conv1_2 = relu(bias_add(
                conv2d(conv1_1, tf.constant(self.vgg16_weights['block1_conv2'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block1_conv2'][1]), name='conv1_2')
            pool1 = max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1')

        with tf.name_scope('conv2'):
            conv2_1 = relu(bias_add(
                conv2d(pool1, tf.constant(self.vgg16_weights['block2_conv1'][0]), strides=[1, 1, 1, 1], padding='SAME'),
                self.vgg16_weights['block2_conv1'][1]), name='conv2_1')
            conv2_2 = relu(bias_add(
                conv2d(conv2_1, tf.constant(self.vgg16_weights['block2_conv2'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block2_conv2'][1]), name='conv2_2')
            pool2 = max_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool2')

        with tf.name_scope('conv3'):
            conv3_1 = relu(bias_add(
                conv2d(pool2, tf.constant(self.vgg16_weights['block3_conv1'][0]), strides=[1, 1, 1, 1], padding='SAME'),
                self.vgg16_weights['block3_conv1'][1]), name='conv3_1')
            conv3_2 = relu(bias_add(
                conv2d(conv3_1, tf.constant(self.vgg16_weights['block3_conv2'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block3_conv2'][1]), name='conv3_2')
            conv3_3 = relu(bias_add(
                conv2d(conv3_2, tf.constant(self.vgg16_weights['block3_conv3'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block3_conv3'][1]), name='conv3_3')
            pool3 = max_pool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool3')

        with tf.name_scope('conv4'):
            conv4_1 = relu(bias_add(
                conv2d(pool3, tf.constant(self.vgg16_weights['block4_conv1'][0]), strides=[1, 1, 1, 1], padding='SAME'),
                self.vgg16_weights['block4_conv1'][1]), name='conv4_1')
            conv4_2 = relu(bias_add(
                conv2d(conv4_1, tf.constant(self.vgg16_weights['block4_conv2'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block4_conv2'][1]), name='conv4_2')
            conv4_3 = relu(bias_add(
                conv2d(conv4_2, tf.constant(self.vgg16_weights['block4_conv3'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block4_conv3'][1]), name='conv4_3')
            pool4 = max_pool(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool4')

        with tf.name_scope('conv5'):
            conv5_1 = relu(bias_add(
                conv2d(pool4, tf.constant(self.vgg16_weights['block5_conv1'][0]), strides=[1, 1, 1, 1], padding='SAME'),
                self.vgg16_weights['block5_conv1'][1]), name='conv5_1')
            conv5_2 = relu(bias_add(
                conv2d(conv5_1, tf.constant(self.vgg16_weights['block5_conv2'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block5_conv2'][1]), name='conv5_2')
            conv5_3 = relu(bias_add(
                conv2d(conv5_2, tf.constant(self.vgg16_weights['block5_conv3'][0]), strides=[1, 1, 1, 1],
                       padding='SAME'), self.vgg16_weights['block5_conv3'][1]), name='conv5_3')
            pool5 = max_pool(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool5')

        c_loss = self.content_loss([input_tensor])
        s_loss = self.style_loss([conv1_1, conv2_1, conv3_1, conv4_1, conv5_1])

        self.loss = self.content_weight * c_loss + self.style_weight * s_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # if self.graph_write:
        #     writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
        #     writer.flush()
        #     writer.close()

    def content_loss(self, conv):
        c_loss_ = 0.0
        for conv_output in conv:
            c_loss_ += l2_loss(conv_output[0] - conv_output[2])
        return c_loss_

    def style_loss(self, conv):
        s_loss_ = 0.0
        for conv_output in conv:
            s_loss_ += l2_loss(self.Gram_matrix(conv_output[1]) - self.Gram_matrix(conv_output[2])) / (
                    4.0 * tf.square(tf.reduce_prod(tf.cast(tf.shape(conv_output[1]), tf.float32))))
        return s_loss_

    def Gram_matrix(self, img):
        img_transpose = tf.transpose(img, (2, 0, 1))
        features = tf.reshape(img_transpose, (tf.shape(img_transpose)[0], -1))
        return tf.matmul(features, tf.transpose(features))

    def stylize(self, epochs, content_wh, magical_image_save_name):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print('Value: ', v)

        for epoch in range(1, epochs + 1):
            loss_, _, magical_img_ = sess.run([self.loss, self.train_op, self.magical_img])
            print('Epoch {:d}/{:d} Loss:{:f}'.format(epoch, epochs, loss_))

            if epoch % 10 == 0:
                imsave(magical_image_save_name, vgg16_depreprocess(magical_img_, content_wh))


if __name__ == '__main__':
    resize_wh = 200
    content_weight = 0.1
    style_weight = 5.0
    epochs = 500
    lr = 10.0

    vgg16_weights = load_vgg16_weights('vgg16.npy')
    content_image_file = 'cat.jpeg'
    style_image_file = 'tangka.jpg'
    magical_image_save_name = style_image_file.split('/')[-1].split('.')[0] + '_' + \
                              content_image_file.split('/')[-1].split('.')[0] + '.png'
    content_wh, content_img, style_img = load_img(content_image_file, style_image_file)
    content_img_value, style_img_value = vgg16_preprocess(content_img, resize_wh), vgg16_preprocess(style_img,
                                                                                                    resize_wh)

    IST = ImageStyleTransfer(vgg16_weights, content_img_value, style_img_value, lr, content_weight, style_weight, True)

    IST.stylize(epochs, content_wh, magical_image_save_name)
