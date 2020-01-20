The model as described in the paper, adapted for the CIFAR10 dataset. Hyperparameters were experimentally found to produce reasonable results.

This version omits the global pathway. The new architecture entails the low-level, mid-level, and colourization network only. (No global, fusion, or classification network).

This leads from observing a heatmap-like output from the original network, where the colourization network outputs one solid colour with a gradient-like change towards where the main object of the picture is located. It seems that maybe it sees a "car" and outputs something that resembles ground+sky + a cluster of car-coloured pixels. Though, these car-pixels are still generally the same shape and location as the car in the l channel.
