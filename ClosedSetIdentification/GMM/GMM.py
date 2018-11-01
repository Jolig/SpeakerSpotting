import numpy as np

from sklearn.mixture import GaussianMixture


def train(X, num_mixtures, covar_type):
    GMM = GaussianMixture(n_components = num_mixtures,
                                covariance_type = covar_type)
    GMM.fit(X)

    return GMM


def test(total_labels, test_labels, test_features, GMM_dict):
    scores = []

    for label, model in GMM_dict.items():
        scores.append(model.score_samples(test_features))

    scores = np.array(scores)
    pred_indices = scores.argmax(axis=0)

    pred_labels = []

    for idx in pred_indices:
        pred_labels.append(total_labels[idx])

    pred_labels = np.array(pred_labels)

    return pred_labels, get_accuracy(test_labels, np.array(pred_labels))


def get_accuracy(pred_labels, test_labels):
    common_list = [i for i, j in zip(test_labels, pred_labels) if i == j]
    cmp_perct = (len(common_list) / len(pred_labels)) * 100

    return cmp_perct


def GMM(total_labels, train_dict, test_labels, test_features, num_mixtures):
    print("\nTraining GMM with "+ str(num_mixtures) +" mixtures...")

    GMM_dict = {key: [] for key in total_labels}

    for label, features in train_dict.items():
        GMM_model = train(features, num_mixtures, "diag")
        GMM_dict[label] = GMM_model

    pred_labels, accuracy = test(total_labels, test_labels, test_features, GMM_dict)
    print("GMM Accuracy : ", accuracy)

    return pred_labels