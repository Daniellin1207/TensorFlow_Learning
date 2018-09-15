# _*_ coding: utf-8 _*_
# @Time : 2018/9/15 下午2:31
# @Author :Daniel
# @File : mnist_test.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist/",one_hot=True)

inputNode=784
outputNode=10

layer1_Node=500
batch=10

x=tf.placeholder(tf.float32,[None,inputNode],name='x_input')

y_=tf.placeholder(tf.float32,[None,outputNode],name='y_input')

weights1=tf.Variable(tf.truncated_normal([inputNode,layer1_Node],stddev=0.1))
biases1=tf.Variable(tf.constant(0.1,shape=[layer1_Node]))

weights2=tf.Variable(tf.truncated_normal([layer1_Node,outputNode],stddev=0.1))
biases2=tf.Variable(tf.constant(0.1,shape=[outputNode]))

layer1=tf.nn.relu(tf.matmul(x,weights1)+biases1)
y=tf.nn.relu(tf.matmul(layer1,weights2)+biases2)

# globals_step=tf.Variable(0,trainable=False)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
cross_entropy_mean=tf.reduce_mean(cross_entropy)

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    validate_feed={x:mnist.train.images,y_:mnist.train.labels}
    test_feed={x:mnist.test.images,y_:mnist.test.labels}

    Step=5000
    for i in range(Step):
        if i%10==0:
            validate_acc=sess.run(accuracy,feed_dict=test_feed)
            print("after %d training steps(s), validation accuracy is %g"%(i,validate_acc))
        xs,ys=mnist.train.next_batch(batch)
        # sess.run(cross_entropy_mean,feed_dict=validate_feed)
        # print("运行{}次".format(i))
        sess.run(train_step,feed_dict={x:xs,y_:ys})
        # print(cross_enftropy_mean)
        print("正运行{}次".format(i))