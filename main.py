from Helper import globalVariables
from Helper import preprocessing

from ClosedSetIdentification.GMM import GMM
from SpeakerVerification.SVM import one_classSVM


def main():
    total_labels, training_dict, test_labels, test_features = \
        preprocessing.loading(globalVariables.data_path, globalVariables.labels_path)

    pred_labels = GMM.GMM(total_labels, training_dict, test_labels, test_features, globalVariables.num_mixtures)

    one_classSVM.oneclassSVM(total_labels, training_dict, test_labels, test_features, pred_labels)


if __name__ == '__main__':
    main()