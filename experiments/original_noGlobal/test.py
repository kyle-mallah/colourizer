from skimage import color
import matplotlib.pyplot as plt
import data_utils

import numpy as np

train, test = data_utils.load_data()

def go(n):
    l = train[0][n]
    ab = train[1][n]

    l = l * 100
    ab = ab*200 - 100

    lab = np.concatenate([l,ab], axis=2)

    rgb = color.lab2rgb(lab)

    return l,ab,lab,rgb

def gg(ab, z):
    lab = np.concatenate([np.zeros_like(ab)+z,ab], axis=2)
    g(lab)

def g(rgb):
    plt.imshow(rgb)
    plt.show()


l,ab,lab,rgb = go(12)
