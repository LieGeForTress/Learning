#https://mofanpy.com/tutorials/machine-learning/tensorflow/example2/
#创建数据
#搭建模型
#计算误差
#传播误差
#初始会话
#不断训练

import tensorflow as tf
import numpy as np

#创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
#定义神经元可变参数
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
#计算误差
loss = tf.reduce_mean(tf.square(y - y_data))
#采用梯度下降法反向传播误差
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

#初始化所有的神经元可变参数
init = tf.global_variables_initializer()
#创建会话
sess = tf.Session()
#执行初始化步骤
sess.run(init)

#不断的训练数据，提升网络性能
for i in range(1000):
    sess.run(train)
    if i % 100 == 0:
        print(i,sess.run(W),sess.run(b))
        print("Loss:%.11f" % sess.run(loss))
