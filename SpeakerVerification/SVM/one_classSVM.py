import numpy as np

from sklearn import svm


def compute_accuracy_perct(crctly_predicted_samples, total_samples):

    return (crctly_predicted_samples/total_samples)*100


def get_accuracy(preds, test_labels, crct_label):
    temp = 0

    for pred, test_label in zip(preds, test_labels):
        if(pred == 1 and crct_label == test_label):
            temp = temp + 1

    return temp


def train(X, n, k):
    svc = svm.OneClassSVM(nu = n, kernel = k)
    svc.fit(X)

    return svc


def test(test_labels, test_features, oneclassSVM_dict):
    crct = 0

    for label, model in oneclassSVM_dict.items():
        crct = crct + get_accuracy(model.predict(test_features), test_labels, label)

    return compute_accuracy_perct(crct, test_features.shape[0])


def predict(test_features, oneclassSVM_dict, pred_labels):
    crct = 0

    for test_feature, pred_label in zip(test_features, pred_labels):
        pred = oneclassSVM_dict[pred_label].predict(np.reshape(test_feature, (1, -1)))
        if(pred == 1):
            crct = crct + 1

    return compute_accuracy_perct(crct, test_features.shape[0])


def oneclassSVM(total_labels, train_dict, test_labels, test_features, pred_labels):
    print("\nTraining One Class SVM...")

    oneclassSVM_dict = {key: [] for key in total_labels}

    for label, features in train_dict.items():
        svc_model = train(features, 0.6, 'rbf')
        oneclassSVM_dict[label] = svc_model

    oneclassSVM_accuracy = test(test_labels, test_features, oneclassSVM_dict)
    GMM_oneclassSVM_accuracy = predict(test_features, oneclassSVM_dict, pred_labels)

    print("One Class SVM Accuracy: ", oneclassSVM_accuracy)
    print("\nOSI(GMM + One Class SVM) Accuracy : ", GMM_oneclassSVM_accuracy)
