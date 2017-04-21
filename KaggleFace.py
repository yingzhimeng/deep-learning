#coding=utf-8
# Here are the links to the data (github has an upload limit)
# https://www.kaggle.com/c/facial-keypoints-detection/data


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as random
# %matplotlib inline

data = pd.read_csv('/home/jiafangliu/学习/bigdata/practices/training/training.csv')
print('data loaded')

# misc variables and sizes of data
images_count = data.iloc[0:, 30:].shape[0]  # images_count = 7049
image_strings = data.iloc[0:, 30:].values
labels = data.iloc[:, 0:30].values
image_array_size = len(image_strings[0, 0].split(' '))  # image_array_size = 9216
desired_locations_count = 30
image_width = image_height = int(np.sqrt(image_array_size))  # image height/width = 96.0
batch_size = 50


# get a batch of images and labels
def get_batch(batch_size):
    tmpImages = []
    tmpLabels = []
    for i in range(0, batch_size):
        rndInt = random.randint(0, batch_size)
        image = image_strings[rndInt, 0].split(' ')
        image = [float(i) for i in image]
        image = [i * (1 / 255) for i in image]
        tmpImages.append(image)
        tmpLabels.append(labels[rndInt])
    return (np.asarray(tmpImages), np.asarray(tmpLabels))


print('made an images tensor as well as a labels tensor for all images in batch')






# set up our weights (or kernals?) and biases for each pixel
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1,1,1,1], 'SAME')

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# placeholder variables
# images
x = tf.placeholder(tf.float32, shape=[None, image_array_size])

# labels
y_ = tf.placeholder(tf.float32, shape=[None, desired_locations_count])

print('methods defined')





# ----------------first convolutional layer---------------------------
W_conv1 = weight_variable([15, 15, 1, 32])
b_conv1 = bias_variable([32])

# turn shape(images_count,9216)  into   (?,96,96,1) ... so it will work with the conv2d method
image = tf.reshape(x, [-1,image_width , image_height,1])
# print (image.get_shape()) # =>(?,96,96,1)

h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
# print (h_conv1.get_shape()) # => (?, 96, 96, 32)

h_pool1 = max_pool_2x2(h_conv1)
# print (h_pool1.get_shape()) # => (?, 48, 48, 32)




#-------------------------- second convolutional layer-----------------
W_conv2 = weight_variable([15, 15, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# print (h_conv2.get_shape()) # => (?, 48,48, 64)
h_pool2 = max_pool_2x2(h_conv2)
# print (h_pool2.get_shape()) # => (?, 24, 24, 64)



#-------------------------fully connected layer-----------------------
W_fc1 = weight_variable([24 * 24 * 64, 1024])
b_fc1 = bias_variable([1024])

# (?, 24, 24, 64) => (?, 36864)
h_pool2_flat = tf.reshape(h_pool2, [-1, 24*24*64])
# print h_pool2_flat.get_shape() = (?, 36864)

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# print (h_fc1.get_shape()) # => (?, 1024)



# ---------------------------------readout layer-------------------------------
W_fc2 = weight_variable([1024, desired_locations_count])
b_fc2 = bias_variable([desired_locations_count])

readout_layer = (tf.matmul(h_fc1, W_fc2) + b_fc2)
# print (readout_layer.get_shape()) #=> (?, 30)

print('Convolutional Nueral Net Defined')








prediction_delta = tf.reduce_sum(tf.abs(readout_layer - y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(prediction_delta)
# train_step = tf.train.GradientDescentOptimizer(.5).minimize(prediction_delta)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
from sklearn import metrics
for i in range(0,100):
    xx, yy = get_batch(100)
    sess.run(train_step, feed_dict={x: xx, y_: yy})

    correct_prediction = tf.equal(tf.argmax(readout_layer, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print (sess.run(accuracy, feed_dict={x: xx, y_: yy}))

    # print(metrics.accuracy_score(yy[0,0], sess.run(readout_layer[0,0], feed_dict={x: xx, y_: yy})))

    # if(i%100 == 0):
    #     print('prediction_delta #' + str(i+1) + ': ' + str(sess.run(prediction_delta, feed_dict={x: xx, y_: yy})))
    #     print('first five labels: ' + str(yy[0,0:5]))
    #     print('first five predictions: ' + str(sess.run(readout_layer[0,0:5], feed_dict={x: xx, y_: yy})))

