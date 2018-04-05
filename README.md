# DeViSE
In this project we go through in detail how we put Deep
Visual-Semantic Embedding Model into practice, and the
ideas we pop up to improve the model.

# Dataset for visual model pre-training
Cifar 100 is a 100 classes image datasets containing
32x32 pixels for each image.
Link :
https://www.cs.toronto.edu/ kriz/cifar-100-python.tar.gz

# How to import data
After import tensorflow, one can include data through comment
below.
tf.keras.datasets.cifar100.load_data

# NETWORK STRUCTURE
In DeViSE paper mentions that one can retrieve pre-train
visual model’s tensors for each image from the layer before
softmax layer which is global max pooling layer. As previous
section saying, we got roughly 43 percents accuracy.
The problem is the mapping is from 64-D to 500-D(the dimension
of text embedding), and the information of images
are so severely compressed that the model can’t learn the
mapping well. Thus we pop up an idea that we retrieve retrain
visual model’s tensors for each image from the layer
before global max pooling which is activation layer, and
the dimension of this layer is 2048.
The result of this improve model improve the accuracy significantly.
The accuracy is roughly 64 percents compared
to the baseline 43 percents accuracy. Besides, we have
tried getting tensors from the layer before activation layer
which is batch normalization layer, but the accuracy drop
to roughly 32 percents.


