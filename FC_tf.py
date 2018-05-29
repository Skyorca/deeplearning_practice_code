import tensorflow as tf
import numpy as np
from numpy.random import RandomState
#Here is a built-in 3-layer FC with tf as practice
"""
how tf works?
1. create constant tensor(e.g.: use for hparam)
tensor looks like this: [[]](1D) [[],[]](2D)...
tf.constant(self-defined fixed tensor, and type(eg. tf.int16, should be defined following tensor))
tf.zeros/ones([shape])
tf.random...     just like numpy

2. create variable tensor(e.g.: use for weight&bias)
tf.Variable(init_tensor, type), value could change but shape can not, initial could be any value

3. create placeholder(e.g.: use for feeding batch data, repeatedly use)
tf.placeholder(type, shape=())
before training(session begins), everything you do within functions are done by placeholders&constants, no variables

4. define operations(from basic operations to complicated functions)
tf.func_name(params)  [some are like numpy]
sess.run(defined_operation, feed_dict)

5. everything starts with a graph
tf.Graph(), used in session. if there is only one graph, without defining a graph is allowed
if you like, use graph like this:
    with graph.as_default():
        define nodes&vects for Net

6. training starts with a session
after having defined NetStructure&operations, start a session to start training.
with tf.Session() as session: #if multiple graphs, tf.Session(graph=graph_x)
    training...
    sess.run(operation, feed_dict)

7. practical notes:
a. use None in shape=(), could alternatively change
b. use tf functions to build loss_function, and define train_step with optimizer(optimizing this loss)
c. how to use Class to make a Net???
d.
tf.reduce_mean: calculate average along whole tensor/appointed axis
a=[[1,1], [2,2]]
tf.reduce_mean(a) = 1.5 ,type=float32
tf.reduce_mean(a, 0) = [1.5, 1.5], do along each column (1+2)/2
tf.reduce_mean(a, 1) = [1, 2], do along each row  1+1/2, 2+2/2

"""

graph = tf.Graph()
#define Net Structure
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=(2, None)) #x=(a,b)
    y = tf.placeholder(tf.float32, shape=(1, None))
    W1 = tf.Variable(tf.random_normal([3,2], stddev=1, seed=1)) # auto. float32
    W2 = tf.Variable(tf.random_normal([1,3], stddev=1, seed=1))
    b1 = tf.Variable(tf.random_normal([1,1]))
    b2 = tf.Variable(tf.random_normal([1,1]))
    #1st layer is input a_0=x
    #2nd layer
    a_1 = tf.nn.relu(tf.matmul(W1, x)+b1) # a_1 is (3, none)
    # output layer
    y_pred = tf.nn.sigmoid(tf.matmul(W2, a_1)+b1) # a_2 is(1, none)
    #define loss
    cross_entropy_loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(y_pred, 1e-10,1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_loss)

#define a simulated sample
rdm=RandomState(1)
data_size=512
X = []
for i in range(data_size):
    X.append(rdm.rand(2,1))
X = np.array(X)

Y = []
for i in range(data_size):
    Y.append([int((X[i][0]+X[i][1])<1)])
Y = np.array(Y)
#begin training
epoch = 100
batch_size = 10
with tf.Session(graph=graph) as sess:
    sess.run(tf.initialize_all_variables())
    for time in range(epoch):
        for i in range(data_size/batch_size+1):
            start = i*batch_size % data_size
            end = min(start+batch_size, batch_size) # build batch
            Y_u = Y.T
            sess.run(train_step, feed_dict={x:X[start:end], y:Y_u[start:end]})
            """
            here is a trick: in 1st batch, start=0, end=10, there are 11 samples;
            However, in slice[start:end], get idx 0~9 actually
            """
        loss = sess.run(cross_entropy_loss, feed_dict={x:X, y:Y})
        print("at %d epoch, loss->%f" % (time, float(loss)))
