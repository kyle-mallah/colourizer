from __future__ import print_function
from __future__ import division

import logging

from skimage import color
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import colourizer
import data_utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('save_dir', './checkpoints/', 
    'directory to load model from')

def add_image_to_axes(ax1, ax2, ax3, l, ab_actual, ab_pred, fix_scale=True):
    if fix_scale:
        l = l * 100
        ab_actual = ab_actual * 200 - 100
        ab_pred = ab_pred * 200 - 100

    image_actual = np.concatenate([l,ab_actual], axis=2)
    image_actual = color.lab2rgb(image_actual)

    image_pred = np.concatenate([l,ab_pred], axis=2)
    image_pred = color.lab2rgb(image_pred)

    ax1.imshow(l[:,:,0]*255, cmap='gray')
    ax2.imshow(image_actual)
    ax3.imshow(image_pred)

def display_5_images(sess, colourizer, test_l, test_ab_actual):
    indices = np.random.choice(
            xrange(len(test_l)), 
            size=5,
            replace=False)

    ls = test_l[indices]
    ab_actuals = test_ab_actual[indices]
    ab_preds = colourizer.predict_colour(sess, ls)

    fig, ax = plt.subplots(5,3)

    for i in xrange(5):
        axes = ax[i]
        ll = ls[i]
        ab_actual = ab_actuals[i]
        ab_pred = ab_preds[i]

        add_image_to_axes(axes[0], axes[1],axes[2], ll, ab_actual, ab_pred)

    plt.show()

def main(unused_argv):
    global colourizer
    
    sess = tf.Session()

    print("Setting up model graph...", end='')
    colourizer = colourizer.Colourizer()
    colourizer.setup()
    print("done")

    print("Loading saver...", end='')
    saver = tf.train.Saver(max_to_keep=5)
    ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
    print("done")

    print("Restoring model...", end='')
    logging.info('Restoring model from %s', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("done")

    print("Loading data...", end='')
    _, (test_l, test_ab, _) = data_utils.load_data()
    print("done")

    while True:
        print("Choose an option (E.g., Enter '1'):")
        print("\t(1) Colourize 5 random novel images")

        user_input = raw_input()

        if user_input == "1":
            display_5_images(sess, colourizer, test_l, test_ab)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()

