import tensorflow as tf
import numpy as np


# x1 = tf.placeholder(tf.float64, shape=(2, 3))
# x2 = [[1.0,2.0,3.0],[4.5,3.5,2.5]]
# x2 = np.array(x2).transpose()

# y = tf.matrix_diag_part(tf.matmul(x1, x2))


# with tf.Session() as sess:
# 	#print(sess.run(y))  # ERROR: will fail because x was not fed.

# 	rand_array = [[1.0,2.0,3.0],[4.5,3.5,2.5]]
	
# 	print(sess.run(y, feed_dict={x1: rand_array}))  # Will succeed.


x1 = tf.placeholder(tf.float64, shape=(2, 3))
x2 = [[1.0,2.0,3.0],[4.5,3.5,2.5]]
x2 = np.array(x2)

# ----- This line is very very very important ------
y = tf.reduce_sum(tf.multiply(x1, x2), 1)
# ----- This line is very very very important ------
# y = tf.multiply(x1, x2)


with tf.Session() as sess:
	#print(sess.run(y))  # ERROR: will fail because x was not fed.

	rand_array = [[1.0,2.0,3.0],[4.5,3.5,2.5]]
	
	print(sess.run(y, feed_dict={x1: rand_array}))  # Will succeed.
