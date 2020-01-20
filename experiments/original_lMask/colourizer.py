from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from skimage import color
import numpy as np

import data_utils

class Colourizer(object):

    def __init__(self, alpha=0.03):
        self.alpha = alpha
        self.global_step = tf.train.get_or_create_global_step()

        self.matrix_init = tf.glorot_normal_initializer()
        self.zeros_init = tf.constant_initializer(0.)

    def _conv_layer(self, x, name, ksize, stride, num_in,
                    num_out, activation=tf.nn.relu,
                    use_bn=True, use_bn_scaling=False):
        """ Build a convolutional layer

        Args:
            x: Tensor of size (b, w, h, num_in) that this layer acts on
            name:The base-name used for any variables created in this layer
            ksize: An integer corresponding to the kernel size
            stride: An integer corresponding to the stride
            num_in: The number of channels in x
            num_out: The number of feature maps to compute
            activation: The activation function to use
            use_bn: Whether to apply batch normalization in this layer
            use_bn_scaling: Whether to using scaling/gamma in batch normalization

        Returns:
            A tensor (b, w/stride, h/stride, num_out) resulting from
            this convolutional layer

        """

        strides = [1, stride, stride, 1]

        # Parameters
        weights = tf.get_variable(name=name+'_w',
            shape=[ksize, ksize, num_in, num_out],
            initializer=self.matrix_init)

        biases = tf.get_variable(name=name+'_biases',
            shape=[num_out],
            initializer=self.zeros_init)

        # Layer
        conv = tf.nn.bias_add(
            tf.nn.conv2d(x, filter=weights, strides=strides, padding='SAME'),
            biases)

        if use_bn:
            conv = batch_norm(conv,
                is_training=self.is_training_, scale=use_bn_scaling)

        if activation:
            conv = activation(conv)

        return conv
    def _dense_layer(self, x, name, num_in,
                    num_out, activation=tf.nn.relu,
                    use_bn=True, use_bn_scaling=False):
        """ Build a dense layer

        Args:
            x: Tensor of size (b, num_in) that is input into this layer
            name: The base-name used for any variables created in this layer
            num_in: The input dimension of this layer
            num_out: The output dimension of this layer
            activation: The activation function to use
            use_bn: Whether to apply batch normalization in this layer
            use_bn_scaling: Whether to using scaling/gamma in batch normalization

        Returns:
            A tensor (b, num_out) resulting from this dense layer

        """

        # Parameters
        weights = tf.get_variable(name=name+'_w',
            shape=[num_in, num_out],
            initializer=self.matrix_init)

        biases = tf.get_variable(name=name+'_b',
            shape=[num_out],
            initializer=self.zeros_init)

        # Layer
        dense = tf.nn.bias_add(tf.matmul(x, weights), biases)

        if use_bn:
            dense = batch_norm(dense,
                is_training=self.is_training_, scale=use_bn_scaling)

        if activation:
            dense = activation(dense)

        return dense

    def _low_network(self, x):
        # First conv layer. 3x3 filter with stride 1. 32 feature maps
        # Output shape: (b, 32, 32, 32)
        conv1 = self._conv_layer(x,
            name='low_conv1', ksize=3, stride=1, num_in=1, num_out=32)

        # Second conv layer. 3x3 filter with stride 1. 64 feature maps
        # Output shape: (b, 32, 32, 64)
        conv2 = self._conv_layer(conv1,
            name='low_conv2', ksize=3, stride=1, num_in=32, num_out=64)

        # Third conv layer. 3x3 filter with stride 2. 64 feature maps
        # Output shape: (b, 16, 16, 64)
        conv3 = self._conv_layer(conv2,
            name='low_conv3', ksize=3, stride=2, num_in=64, num_out=64)

        # Fourth conv layer. 3x3 filter with stride 1. 128 feature maps
        # Output shape: (b, 16, 16, 128)
        conv4 = self._conv_layer(conv3,
            name='low_conv4', ksize=3, stride=1, num_in=64, num_out=128)

        return conv4

    def _mid_network(self, x):
        # First conv layer. 3x3 filter with stride 2. 128 feature maps
        # Output shape: (b, 8, 8, 128)
        conv1 = self._conv_layer(x,
            name='mid_conv1', ksize=3, stride=2, num_in=128, num_out=128)

        # Second conv layer. 3x3 filter with stride 1. 64 feature maps
        # Output shape: (b, 8, 8, 64)
        conv2 = self._conv_layer(conv1,
            name='mid_conv2', ksize=3, stride=1, num_in=128, num_out=64)

        return conv2


    def _global_network(self, x):
        # First conv layer. 3x3 filter with stride 2. 128 feature maps
        # Output shape: (b, 8, 8, 128)
        conv1 = self._conv_layer(x,
            name='global_conv1', ksize=3, stride=2, num_in=128, num_out=128)

        # Second conv layer. 3x3 filter with stride 1. 128 feature maps
        # Output shape: (b, 8, 8, 128)
        conv2 = self._conv_layer(conv1,
            name='global_conv2', ksize=3, stride=1, num_in=128, num_out=128)

        # Third conv layer. 3x3 filter with stride 2. 128 feature maps
        # Output shape: (b, 4, 4, 128)
        conv3 = self._conv_layer(conv2,
            name='global_conv3', ksize=3, stride=2, num_in=128, num_out=128)

        # Second conv layer. 3x3 filter with stride 1. 64 feature maps
        # Output shape: (b, 4, 4, 128)
        conv4 = self._conv_layer(conv3,
            name='global_conv4', ksize=3, stride=1, num_in=128, num_out=128)


        conv4_flat = tf.reshape(conv4, [-1, 4*4*128])
        dense1 = self._dense_layer(conv4_flat,
            name='global_dense1', num_in=4*4*128, num_out=512)

        dense2 = self._dense_layer(dense1,
            name='global_dense2', num_in=512, num_out=128)

        dense3 = self._dense_layer(dense2,
            name='global_dense3', num_in=128, num_out=64)

        return dense2, dense3


    def _fusion_network(self, dense, conv):
        """ Implementation of the "fusion layer" as described in the paper

        Args:
            dense: tensor of shape (1,64)
                from the global-features network
            conv: tensor of shape (b,8,8,64)
                from the mid-features network

        Returns:
            Tensor of shape (b, 8, 8, 32) of the
            fused global and mid-level features

        """

        # Fusion parameters
        fusion_weights = tf.get_variable(name='fusion_w',
            shape=[64, 128],
            initializer=self.matrix_init)

        fusion_biases = tf.get_variable(name='fusion_b',
            shape=[64],
            initializer=self.zeros_init)

        # Fusion layer

        # Replicate dense vectors 8x8 times creating a volume of size bx8x8x64
        tiled = tf.tile(dense, [8,8])
        tiled = tf.reshape(tiled, (tf.shape(conv)[0],8,8,64))

        fusion_pre = tf.concat(
            [tiled, conv],
            axis=3)
        fusion_pre = tf.nn.bias_add(
            tf.einsum('bijk,lk->bijl', fusion_pre, fusion_weights),
            fusion_biases)
        fusion_bn = batch_norm(fusion_pre, is_training=self.is_training_, scale=False)
        fusion = tf.nn.relu(fusion_bn)

        # First conv layer. 3x3 filter with stride 1. 32 feature maps
        # Output shape: (b, 8, 8, 64)
        fusion_conv = self._conv_layer(fusion,
            name='fusion_conv1', ksize=3, stride=1, num_in=64, num_out=64)

        return fusion_conv

    def _colorization_network(self, x, l_mask):
        # First upsample: using nearest neighbours interpolation
        # (b, 8, 8, 64) --> (b, 16, 16, 64)
        upsample1 = tf.image.resize_images(x, size=[16, 16],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # First conv layer. 3x3 filter with stride 1. 32 feature maps
        # Output shape: (b, 16, 16, 64)
        conv1 = self._conv_layer(upsample1,
            name='colour_conv1', ksize=3, stride=1, num_in=64, num_out=32)

        # Second conv layer. 3x3 filter with stride 1. 16 feature maps
        # Output shape: (b, 16, 16, 32)
        conv2 = self._conv_layer(conv1,
            name='colour_conv2', ksize=3, stride=1, num_in=32, num_out=16)

        # Second upsample: using nearest neighbours interpolation
        # (b, 16, 16, 64) --> (b, 32, 32, 16)
        upsample2 = tf.image.resize_images(conv2, size=[32, 32],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Third conv layer. 3x3 filter with stride 1. 16 feature maps
        # Output shape: (b, 32, 32, 8)
        conv3 = self._conv_layer(upsample2,
            name='colour_conv3', ksize=3, stride=1, num_in=16, num_out=8)

        # Fourth conv layer. 3x3 filter with stride 1. 16 feature maps
        # Output shape: (b, 32, 32, 2)
        conv4 = self._conv_layer(conv3,
            name='colour_output', ksize=3, stride=1, num_in=8, num_out=2,
            activation=None, use_bn_scaling=True)

        output = tf.sigmoid(conv4 * l_mask)

        return output

    def _classifier_network(self, x):
        dense1 = self._dense_layer(x,
            name='classifier_dense1',
            num_in=128,
            num_out=64)

        dense2 = self._dense_layer(dense1,
            name='classifier_dense2',
            num_in=64,
            num_out=10,
            activation=None,
            use_bn=False)

        # Softmax on output applied by cost function
        return dense2


    def _core_builder(self):
        """Builds the tensorflow graph for the model

        Returns:
            loss and colour-prediction ops

        """

        with tf.variable_scope("core", reuse=tf.AUTO_REUSE):
            low_features = self._low_network(self.l_)
            mid_features = self._mid_network(low_features)

            classifier_features, global_features = self._global_network(low_features)

            fused_features = self._fusion_network(global_features, mid_features)
            ab_predictions = self._colorization_network(fused_features, self.l_)
            colour_loss = tf.losses.mean_squared_error(
                labels=self.ab_,
                predictions=ab_predictions)

            classifier_logits = self._classifier_network(classifier_features)
            classifier_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=classifier_logits,
                    labels=self.y_))

            # Final loss
            total_loss = colour_loss + self.alpha*classifier_loss

            classifier_softmax = tf.nn.softmax(classifier_logits)
            y_pred = tf.argmax(classifier_softmax, axis=1)
            y_actual = tf.argmax(self.y_, axis=1)
            classifier_accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(y_pred, y_actual),
                    tf.float32))

            # image_pred = tf.concat([self.l_,ab_predictions], axis=3)
            # image_actual = tf.concat([self.l_, self.ab_], axis=3)
            # image_compared = tf.concat([image_actual, image_pred], axis=2)

            z = tf.zeros_like(self.l_) + 0.15
            image_actual = tf.concat([z, self.ab_], axis=3)
            image_pred = tf.concat([z,ab_predictions], axis=3)
            image_l = tf.tile(self.l_, [1,1,1,3])

            image_compared = tf.concat([image_l, image_actual, image_pred], axis=2)

            with tf.name_scope('summary'):
                tf.summary.image("image_compared", image_compared, max_outputs=5)

                tf.summary.scalar("colour_loss", tf.reduce_mean(colour_loss))
                tf.summary.scalar("classifier_loss", tf.reduce_mean(classifier_loss))

                tf.summary.histogram("classifier_softmax", classifier_softmax)
                tf.summary.scalar("classifier_accuracy", classifier_accuracy)

        return total_loss, ab_predictions

    def _get_optimizer(self):
        """
        Get the optimizer used for training
        """
        return tf.train.AdamOptimizer(0.001,
                                      epsilon=1e-4)

    def _get_placeholders(self):
        """Get all placeholders for training

        Returns:
                x:  placeholder for l component of images
                ab: placeholder for ab components of images
                y:  placeholder for the corresponding labels of images
                is_training: boolean placeholder indicating whether we
                    are training or not
        """

        return (tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='x'),
                tf.placeholder(tf.float32, shape=[None, 32, 32, 2], name='ab'),
                tf.placeholder(tf.float32, shape=[None, 10], name='y'),
                tf.placeholder(tf.bool, name='is_training'))

    def run_global_step(self, sess):
        return sess.run(self.global_step)

    def setup(self):
        self.l_, self.ab_, self.y_, self.is_training_ = self._get_placeholders()

        ops = self._core_builder()
        self.loss, self.ab_predictions = ops

        optimizer = self._get_optimizer()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(
                self.loss,
                global_step=self.global_step)

        self.merged_summaries = tf.summary.merge_all()

    def predict_colour(self, sess, l):
        """ Predict the ab components of greyscale images l
        Args:
            sess: A Tensorflow Session
            l: A tensor (n, 32, 32, 1) of n 32x32 greyscale images
               corresponding to the l component in Lab colour space

        Returns:
            A tensor (n, 32, 32, 2) of predicted ab components

        """

        return sess.run(
            self.ab_predictions,
            feed_dict={self.l_:l, self.is_training_:False})

    def predict_class(self, sess, l):
        """ Predict the class that each image in l belongs to
        Args:
            sess: A Tensorflow Session
            l: A tensor (n, 32, 32, 1) of n 32x32 greyscale images
               corresponding to the l component in Lab colour space

        Returns:
            A tensor of (n, 1) corresponding to the n predicted class labels
        """

        return sess.run(
            self.class_predictions,
            feed_dict={self.l_:l, self.is_training_:False})

    def get_loss(self, sess, l, ab, y):
        """ Compute the total loss based on l, ab, y
        Args:
            sess: A Tensorflow Session
            l: A tensor (b, 32, 32, 1) of n 32x32 greyscale images
               corresponding to the l component in Lab colour space

        Returns:
            A tensor of (n, 1) corresponding to the n predicted class labels
        """
        return sess.run(
            self.loss,
            feed_dict={self.l_:l,
                        self.ab_:ab,
                        self.y_:y,
                        self.is_training_:False})

    def one_step(self, sess, l, ab, y, summarize=False):
        """ Train for one batch
        Args:
            sess: A Tensorflow Session
            l: A list of batches of l components of images
            ab: A list of batches of ab components of images
            y: A list of batches of labels that correspond to each image

        Returns:
            The loss for this batch

        """
        outputs = [self.loss, self.train_step]
        if summarize:
            outputs.append(self.merged_summaries)

        out = sess.run(
            outputs,
            feed_dict={self.l_:l,
                        self.ab_:ab,
                        self.y_:y,
                        self.is_training_:True})
        if summarize:
            return out[0], out[2] # loss, summary
        else:
            return out[0] # loss

    # def epoch_step(self, sess, l, ab, y):
    #     """ Train for one epoch
    #     Args:
    #         sess: A Tensorflow Session
    #         l: A list of batches of l components of images
    #         ab: A list of batches of ab components of images
    #         y: A list of batches of labels that correspond to each image
    #
    #     Returns:
    #         A List of losses for all the batches
    #
    #     """
    #
    #     outputs = [self.loss, self.train_step]
    #     losses = []
    #
    #     iterator = zip(l,ab,y)
    #     for ll,abab,yy in iterator:
    #         out = sess.run(
    #             outputs,
    #             feed_dict={self.l_:ll,
    #                         self.ab_:abab,
    #                         self.y_:yy,
    #                         self.is_training_:True})
    #         losses.append(out[0])
    #
    #     return losses
