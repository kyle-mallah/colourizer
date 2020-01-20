import numpy as np
import tensorflow as tf
from skimage import color
import csv
from scipy.ndimage.filters import gaussian_filter as gaussian_filter
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

# pixel A/B values range between [-110, 110]
def mapToBinIndex(val):
    return int(np.floor((val+110.0)/10.0))

def countPixel(a,b):
    aa = mapToBinIndex(a)
    bb = mapToBinIndex(b)
    bins[aa][bb] += 1

cifar10 = tf.keras.datasets.cifar10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()
train_x = color.rgb2lab(train_x)

train_x = train_x[:,:,:,1:]
bins = [[0 for i in xrange(22)] for _ in xrange(22)]

for example in tqdm(train_x.reshape(-1, 2)):
    countPixel(*example)

bins = np.array(bins, dtype='float64')
bins = bins / np.sum(bins)
print bins

with open("empiricalBinDistribution", 'w') as f:
    np.save(f, bins)

smoothedBins = gaussian_filter(bins, sigma=5)
print smoothedBins

with open("smoothedBinDistribution", 'w') as f:
    np.save(f, smoothedBins)
