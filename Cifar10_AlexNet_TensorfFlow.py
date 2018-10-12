import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn.preprocessing
from util_functions import *

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
y_train=Indices2OneHot(y_train)
y_test=Indices2OneHot(y_test)
x_train.astype(float)
y_train.astype(float)
#x_train=tf.convert_to_tensor(x_train)
#y_train=tf.convert_to_tensor(y_train)
# Hyper parameters
learning_rate = 0.0001
training_iters = 200000
batch_size = 64
display_step = 20

# Parameters
n_classes = 10 # # of class
dropout = 0.8 # Dropout probability

# Placeholder
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3]) # size of cifar10 is 32*32*3
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# Alexnet
pred, para=alex_net(x)

# loss function
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=y))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initialize
init = tf.global_variables_initializer()

# Accuracy
correct_pred = tf.equal(tf.argmax(pred, -1), tf.argmax(y, -1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss_value=10
# # start training
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while loss_value>0.1:
        batch_index = np.random.randint(0, 4999, 64)
        image_batch=x_train[batch_index]
        label_batch=y_train[batch_index]
        sess.run(optimizer, feed_dict={x: image_batch, y: label_batch, keep_prob: dropout})
        if step % 50 == 0:
            # print(sess.run(pred, feed_dict={x: image_batch, y: label_batch, keep_prob: dropout}))
            # accracy, prediction and loss
            accu_value, pred_value, loss_value = sess.run([accuracy, pred, loss], feed_dict={x: image_batch, y: label_batch, keep_prob: 1.})
            print("step: "+ str(step)+"  loss: "+ str(loss_value)+" accuracy: "+ str(accu_value))
            parameters = sess.run(para)
            print(parameters[1][1])
        step += 1
    print("Optimization Finished!")
#     # testing accuracy
#     # print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.})
# with tf.Session() as sess:
#      sess.run(init)
#      batch_index = np.random.randint(0, 4999, 64)
#      image_batch=x_train[batch_index]
#      label_batch=y_train[batch_index]
#      pred_value, loss_value = sess.run([pred, loss], feed_dict={x: image_batch, y: label_batch, keep_prob: 1.})
#      print   "  loss: " + str(loss_value)  # +" accuracy: "+ str(accu_value)
#      print(pred_value)