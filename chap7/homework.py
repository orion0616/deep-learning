def homework(train_X, train_y, test_X):
    import numpy as np
    import tensorflow as tf
    from sklearn.utils import shuffle
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from tensorflow.examples.tutorials.mnist import input_data

    rng = np.random.RandomState(1234)
    random_state = 42

    class Conv:
        def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='VALID'):
            # Xavier Initialization
            fan_in = np.prod(filter_shape[:3]) # when filter_shape=(5,5,1,20), fan_in -> 5 * 5 * 1 = 25
            fan_out = np.prod(filter_shape[:2]) * filter_shape[3] # when filter_shape=(5,5,1,20), fan_out -> (5 * 5) * 20 = 500
            self.W = tf.Variable(rng.uniform(
                            low=-np.sqrt(6/(fan_in + fan_out)),
                            high=np.sqrt(6/(fan_in + fan_out)),
                            size=filter_shape
                        ).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')
            self.function = function
            self.strides = strides
            self.padding = padding

        def f_prop(self, x):
            u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
            return self.function(u)

    class Pooling:
        def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
            self.ksize = ksize
            self.strides = strides
            self.padding = padding
        
        def f_prop(self, x):
            return tf.nn.avg_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)

    class Flatten:
        def f_prop(self, x):
            return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

    class Dense:
        def __init__(self, in_dim, out_dim, function=lambda x: x):
            # Xavier Initialization
            self.W = tf.Variable(rng.uniform(
                            low=-np.sqrt(6/(in_dim + out_dim)),
                            high=np.sqrt(6/(in_dim + out_dim)),
                            size=(in_dim, out_dim)
                        ).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
            self.function = function

        def f_prop(self, x):
            return self.function(tf.matmul(x, self.W) + self.b)

    # Initialize
    print("start initializing")
    layers = [                            # (縦の次元数)x(横の次元数)x(チャネル数)
        Conv((5, 5, 1, 20), tf.nn.relu),  # 28x28x 1 -> 24x24x20
        Pooling((1, 2, 2, 1)),            # 24x24x20 -> 12x12x20
        Conv((5, 5, 20, 50), tf.nn.relu), # 12x12x20 ->  8x 8x50
        Pooling((1, 2, 2, 1)),            #  8x 8x50 ->  4x 4x50
        Flatten(),
        Dense(4*4*50, 10, tf.nn.softmax)
    ]

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    t = tf.placeholder(tf.float32, [None, 10])

    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    y = f_props(layers, x)

    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    valid = tf.argmax(y, 1)

    # training
    print("start training")
    n_epochs = 10
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size

    init = tf.global_variables_initializer()
    print("start session")
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            print(epoch)
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: train_X, t: train_y})

        # apply
        print("start applying")
        z = tf.placeholder(tf.float32, [None, 28, 28, 1])
        result = f_props(layers, z)
        ans = tf.argmax(result, 1)
        pred_y = sess.run(ans, feed_dict={z: test_X})
    return pred_y
