from __future__ import division
from __future__ import print_function

import tensorflow as tf
from skimage import color
import numpy as np

cifar10 = tf.keras.datasets.cifar10
num_classes = 10

def prepare_images(images):
    """ Converts array of RGB images to Lab and scales values to [0, 1]
    
    Args:
        images: (n, w, h, 3) array of n RGB images 

    Return:
        xs: (n, w, h) array of n images, corresponding to the L dimension
        ys: (n, w, h, 2) array of n images, corresponding to the a/b dimensions
        
    """
    images = color.rgb2lab(images)

    l = images[:,:,:,:1]/100.
    ab = images[:,:,:,1:]/200. + 0.5

    return l, ab

def load_data():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    train_l, train_ab = prepare_images(train_x)
    test_l, test_ab = prepare_images(test_x)

    train_y = tf.keras.utils.to_categorical(train_y, num_classes)
    test_y  = tf.keras.utils.to_categorical(test_y, num_classes)

    return (train_l, train_ab, train_y), (test_l, test_ab, test_y)

