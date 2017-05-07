def homework(train_X, train_y, test_X):
    # WRITE ME!
    from sklearn.utils import shuffle
    import numpy as np
    import sys

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def deriv_sigmoid(x):
        return sigmoid(x) *(1-sigmoid(x))

    def softmax(x):
        exp_x = np.exp(x)
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)

    def deriv_softmax(x):
        return softmax(x)*(1 - softmax(x))

    def train(x, t, eps=0.3):
        nonlocal W1,W2,b1,b2
        # Forward Propagation
        u1 = np.matmul(x, W1) + b1
        z1 = sigmoid(u1)
        u2 = np.matmul(z1, W2) + b2
        z2 = softmax(u2)

        # Back Propagation (Cost Function: Negative Loglikelihood)
        y = z2
        cost = np.sum(-t*np.log(y) - (1 - t)*np.log(1 - y))
        delta_2 = y - t
        delta_1 = deriv_sigmoid(u1) * np.matmul(delta_2, W2.T)

        # Update Parameters
        dW1 = np.matmul(x.T, delta_1)
        db1 = np.matmul(np.ones(len(x)), delta_1)
        W1 = W1 - eps*dW1
        b1 = b1 - eps*db1

        dW2 = np.matmul(z1.T, delta_2)
        db2 = np.matmul(np.ones(len(z2)), delta_2)
        W2 = W2 - eps*dW2
        b2 = b2 - eps*db2

        return cost


    def test(x):
        nonlocal W1,W2,b1,b2
        u1 = np.matmul(x, W1) + b1
        z1 = sigmoid(u1)
        u2 = np.matmul(z1, W2) + b2
        z2 = softmax(u2)

        y = np.argmax(z2,axis=1).reshape(100)
        return y

    #initialize
    # Layer1 weights
    W1 = np.random.normal(loc=0,scale=0.1, size=(784, 400))
    b1 = np.zeros(400)
    # Layer2 weights
    W2 = np.random.normal(loc=0,scale=0.1, size=(400, 10))
    b2 = np.zeros(10)

    for epoch in range(100):
        for x,y in zip(train_X, train_y[:,np.newaxis]):
            t = np.zeros(10)
            t[y] = 1
            cost = train(x[np.newaxis, :], t[np.newaxis, :])
    pred_y = test(test_X)
    return pred_y
