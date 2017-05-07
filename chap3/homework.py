def homework(train_X, train_y, test_X):
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold


    def cosDist(v1, v2):
        return np.dot(v1,v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    def kNeiborhood(trainx, trainy, validatex,k):
        from scipy import stats
        predict = np.zeros(len(validatex))
        for i,v in enumerate(validatex):
            distances = np.zeros(len(trainx))
            for j,trainV in enumerate(trainx):
                distances[j] = cosDist(trainV, v)
            index = np.argsort(distances)[::-1]
            sortedy = trainy[index]
            ky = sortedy[0:k]
            result = stats.mode(ky)
            predict[i] = result[0][0]
        return predict

    def determineK(train_X, train_y, test_X):
        from sklearn.metrics import f1_score
        eLambda = []
        for k in range(1,20):
            sumE = 0
            kf = KFold(n_splits=5, shuffle = True)
            for trainIndex, validateIndex in kf.split(train_X):
                trainx = train_X[trainIndex]
                validatex = train_X[validateIndex]
                trainy = train_y[trainIndex]
                validatey = train_y[validateIndex]

                pred = kNeiborhood(trainx, trainy, validatex, k)
                sumE += f1_score(validatey, pred, average="macro")
            eLambda.append(sumE/5)
        maxE = 0 
        K = 1
        for i,e in enumerate(eLambda):
            if maxE < e:
                maxE = e
                K = i+1
        return K

    k = determineK(train_X, train_y, test_X)
    pred_y = kNeiborhood(train_X, train_y, test_X, k)
    return pred_y # pred_y is a label with test_X
