# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

del [
    tf.app,
    tf.compat,
    tf.contrib,
    tf.errors,
    tf.gfile,
    tf.graph_util,
    tf.image,
    tf.layers,
    tf.logging,
    tf.losses,
    tf.metrics,
    tf.python_io,
    tf.resource_loader,
    tf.saved_model,
    tf.sdca,
    tf.sets,
    tf.summary,
    tf.sysconfig,
    tf.test,
    tf.train
]

def homework(train_X, train_y, test_X):
    # WRITE ME!
    import numpy as np
    import tensorflow as tf
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from tensorflow.examples.tutorials.mnist import input_data

    rng = np.random.RandomState(1234)
    random_state = 42

    x = tf.placeholder(tf.float32, name='x')
    t = tf.placeholder(tf.float32, name='t')

    W1 = tf.Variable(rng.normal(loc=0.0, scale = 0.5, size=(784,200)).astype('float32'), name='W1')
    b1 = tf.Variable(np.zeros(200).astype('float32'), name='b1')

    W2 = tf.Variable(rng.normal(loc=0.0, scale = 0.5, size=(200,10)).astype('float32'), name='W2')
    b2 = tf.Variable(np.zeros(10).astype('float32'), name='b2')
    params = [W1, b1, W2, b2]

    z1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    #z1 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(x,W1) + b1), 0.8)
    z2 = tf.nn.softmax(tf.matmul(z1, W2) + b2)
    #z2 = tf.nn.dropout(tf.nn.softmax(tf.matmul(z1,W2) + b2), 0.8)

    y = z2

    cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(tf.clip_by_value(y, 1e-10, 1.0))))

    gW1, gb1, gW2, gb2 = tf.gradients(cost, params)
    updates = [
        W1.assign_add(-0.01*gW1),
        b1.assign_add(-0.01*gb1),
        W2.assign_add(-0.01*gW2),
        b2.assign_add(-0.031*gb2)
    ]
    train = tf.group(*updates)

    valid = tf.argmax(y, 1)


    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_X, mnist_y = mnist.train.images, mnist.train.labels
    train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=random_state)

    n_epochs = 30
    batch_size = 100
    n_batches = train_X.shape[0] // batch_size

# Step5. 学習
    import datetime
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: valid_X, t: valid_y})
            print('EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))
            print(datetime.datetime.today())

        # apply
        realData = tf.placeholder(tf.float32, name='r')
        #realz1 = tf.nn.sigmoid(tf.matmul(realData, W1) + b1) * 0.8
        realz1 = tf.nn.sigmoid(tf.matmul(realData, W1) + b1)
        #realz2 = tf.nn.softmax(tf.matmul(realz1, W2) + b2) * 0.8
        realz2 = tf.nn.softmax(tf.matmul(realz1, W2) + b2)
        solve = tf.argmax(realz2, 1)
        pred_y = sess.run(solve, feed_dict={realData: test_X})
        

    
    return pred_y



def load_mnist():
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                               mnist.target.astype('int32'), random_state=42)

    mnist_X = mnist_X / 255.0

    return train_test_split(mnist_X, mnist_y,
                test_size=0.2,
                random_state=42)

def validate_homework():
    train_X, test_X, train_y, test_y = load_mnist()

    # validate for small dataset
    train_X_mini = train_X[:100]
    train_y_mini = train_y[:100]
    test_X_mini = test_X[:100]
    test_y_mini = test_y[:100]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(test_y_mini, pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(test_y, pred_y, average='macro'))

validate_homework()
