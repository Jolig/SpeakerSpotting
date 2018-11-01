import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def reshaping_dict(dict):
    for label, features in dict.items():
        dict[label] = np.array(dict[label])
        shape = dict[label].shape
        dict[label] = np.reshape(dict[label], (shape[1], shape[2]))

    return dict


def load_labels(labels_path):
    labels = np.array(pd.read_csv(labels_path, sep='\n', header=None, engine='python'))
    numlabels = labels.shape[0]
    total_labels = np.reshape(labels, (numlabels))

    return total_labels


def load_features(labels, data_path):
    train_dict = {key: [] for key in labels}
    test_labels = []
    test_features = []
    numlabels = len(labels)

    for label_idx in range(0, numlabels):
        label = labels[label_idx]
        fileName = data_path + label + '_trn_mfcc.txt'
        features = np.array(pd.read_csv(fileName, sep='\t\t', header=None, engine='python'))

        training_features, testing_features = train_test_split(features, test_size = 0.20, random_state = 42)

        train_dict[label].append(np.array(training_features))
        test_labels.extend([label] * len(testing_features))
        test_features.extend(testing_features)

    return reshaping_dict(train_dict), np.array(test_labels), np.array(test_features)


def loading(data_path, labels_path):
    print("Loading Data...")

    total_labels = load_labels(labels_path)

    train_dict, test_labels, test_features = load_features(total_labels, data_path)

    print("Loading Completed...")

    return total_labels, train_dict, test_labels, test_features
