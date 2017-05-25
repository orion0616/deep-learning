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
        
        # apply
        realData = tf.placeholder(tf.float32, name='r')
        #realz1 = tf.nn.sigmoid(tf.matmul(realData, W1) + b1) * 0.8
        realz1 = tf.nn.sigmoid(tf.matmul(realData, W1) + b1)
        #realz2 = tf.nn.softmax(tf.matmul(realz1, W2) + b2) * 0.8
        realz2 = tf.nn.softmax(tf.matmul(realz1, W2) + b2)
        solve = tf.argmax(realz2, 1)
        pred_y = sess.run(solve, feed_dict={realData: test_X})
        

    
    return pred_y
