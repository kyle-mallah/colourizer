from __future__ import division
from __future__ import print_function

import logging
import os

import colourizer
import data_utils

import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('alpha', 0.003, 'alpha (multiplier for classifier loss)')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_string('save_dir', './checkpoints/',
                        'directory to save model to')
tf.flags.DEFINE_string('summary_dir', './summaries/', 'location to save summary logs')
tf.flags.DEFINE_integer('summary_freq', 100, 'number of batches between each summary log')
tf.flags.DEFINE_integer('test_frequency', 20,
                        'every so many training iterations, '
                        'assess test accuracy')
tf.flags.DEFINE_integer('test_size', 256,
                        'number of images to use to compute '
                        'test accuracy')
tf.flags.DEFINE_integer('num_epochs', 20, 'number of training epochs')

class Trainer(object):

    def __init__(self, model, train_l, train_ab, train_y, test_l, test_ab, test_y):
        assert len(train_l) == len(train_ab) == len(train_y)
        assert len(test_l) == len(test_ab) == len(test_y)

        self.num_train = len(train_l)
        self.num_test = len(test_l)

        self.model = model
        self.train_l  = train_l
        self.train_ab = train_ab
        self.train_y  = train_y
        self.test_l  = test_l
        self.test_ab = test_ab
        self.test_y  = test_y


    def get_epoch(self):
        """ Get batches corresponding to one epoch.
            Training examples are shuffled each time this is called.

        Returns:
            A tuple (l, ab, y) of lists containing all batches for one entire
            epoch.
            Each batch is a numpy array with first dimension of size batch_size
        """

        num_batches = len(self.train_l) // FLAGS.batch_size
        epoch_size = num_batches * FLAGS.batch_size

        indices = np.arange(epoch_size)
        np.random.shuffle(indices)

        l  = np.split(self.train_l[indices],  num_batches)
        ab = np.split(self.train_ab[indices], num_batches)
        y  = np.split(self.train_y[indices],  num_batches)

        return l, ab, y

    def test_accuracy(self, sess, epoch, num_epochs):
        indices = np.random.choice(
            xrange(self.num_test),
            FLAGS.test_size,
            replace=False)
        l = self.test_l[indices]
        ab = self.test_ab[indices]
        y = self.test_y[indices]

        loss = self.model.get_loss(sess, l, ab, y)
        logging.info(
            'Loss (Epoch {0}/{1}): {2}'.format(
                epoch+1, num_epochs, loss))

    def test_classification_accuracy(self, sess, epoch, num_epochs):
        indices = np.random.choice(
            xrange(self.num_test),
            1000,
            replace=False)
        l = self.test_l[indices]
        y = np.argmax(self.test_y[indices], axis=1)

        y_ = self.model.predict_class(sess, l)

        accuracy = np.mean(y == y_)

        logging.info(
            'Classification Accuracy (Epoch {0}/{1}): {2}'.format(
                epoch+1, num_epochs, accuracy))

    def run(self):
        print('running...')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=5)
        ckpt = None
        if FLAGS.save_dir:
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            logging.info('restoring from %s', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # Summary writer
        train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        for epoch in xrange(FLAGS.num_epochs):
            batches = self.get_epoch()

            for batch,(l,ab,y) in enumerate(zip(*batches)):
                self.model.one_step(sess,l,ab,y)

                if batch % FLAGS.summary_freq == 0:
                    loss, summary = self.model.one_step(sess,l,ab,y,summarize=True)
                    train_writer.add_summary(summary, self.model.run_global_step(sess))

            if saver and FLAGS.save_dir:
                saved_file = saver.save(
                    sess,
                    os.path.join(FLAGS.save_dir, 'model.ckpt'),
                    global_step=self.model.global_step)
                logging.info('saved model to {0}'.format(saved_file))

            logging.info(
                'epoch {0}\{1} complete.'.format(
                    epoch+1, FLAGS.num_epochs))

def main(unused_argv):
    model = colourizer.Colourizer(FLAGS.alpha)
    model.setup()

    print("loading data...")
    train, test = data_utils.load_data()
    print("done\n")

    trainer = Trainer(model, *train+test)
    trainer.run()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
