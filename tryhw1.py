from uwnet import *

def conv_net(): #1108480
    l = [   make_connected_layer(32*32*3, 320), #221184
# int w, int h, int c, int filters, int size, int stride
# ((height - 1) / stride + 1) * ((width - 1) / stride + 1) * size^2 * c * filters
# (31/1 + 1) * (31/1 + 1) * 3^2 * 3 * 8
            make_activation_layer(RELU),
            make_connected_layer(320, 256), #294912 #
            make_activation_layer(RELU),
            make_connected_layer(256, 128),# 294912
            make_activation_layer(RELU),
            make_connected_layer(128, 64), #294912 
            make_activation_layer(RELU),
            make_connected_layer(64, 10), # 2560
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How many operations does the convnet use during a forward pass?
# Only count matrix operations

# The covnet performs roughly 1,108,480 operations during a forward pass

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# The covnet had a training accuracy of 0.685, and a testing accuracy of 0.629
# The fully connnected had a training accuracy of 0.562 and a testing accuracy of 0.519
# Therefore the convolutional networks were more efficient (and therefore better) for 
# classification than the fully connected network. This makes sense as tiling the same kernel 
# across the image can better extract features than a fully connnected, which would have to learn
# weights for each section of the image to produce the same results. Since the kernel learns broad
# features, it also generalizes much better.

