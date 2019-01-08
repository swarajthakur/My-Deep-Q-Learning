import time
start_time = time.time()
import tensorflow as tf
#with tf.device('/cpu:0'):
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
timer = time.time()
print(timer - start_time)