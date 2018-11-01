from sklearn import svm


def twoClassSVM(X, y, k):
    svc = svm.SVC(kernel = k,probability = True)
    svc.fit(X, y)

    return svc