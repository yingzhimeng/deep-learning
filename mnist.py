import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# input
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None,10])

# inference
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
matm=tf.matmul(x,W)
y = tf.nn.softmax(tf.matmul(x,W) + b)

# loss
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# training
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# training cycles
sess = tf.Session()
sess.run(tf.global_variables_initializer())
start=time.clock()
for i in range(100):
 batch_xs, batch_ys = mnist.train.next_batch(100)
 sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

 print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

elapsed = (time.clock() - start)
print("Time used:",elapsed)