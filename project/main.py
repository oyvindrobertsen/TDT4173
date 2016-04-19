from random import choice

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.datasets.base import Bunch

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from preprocessing import binarize, oriented_gradients
from read_dataset import DATA_SHAPE, get_dataset
from utils import int_to_letter

"""
Modified from http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py
"""


def load_dataset(n=25):
    # each row of the matrix is x pixel intensity values and 1 value representing the class of the image
    dataset_matrix = get_dataset(n)

    target = dataset_matrix[:, -1]  # gets the last column, classes
    image_data = dataset_matrix[:, :-1]  # gets every column but the last, image data

    images = image_data.view()

    images.shape = DATA_SHAPE

    return Bunch(data=image_data,
                 target=target.astype(np.int),
                 target_names=np.arange(ALPHABET_SIZE),
                 images=images,
                 DESCR='')


def split_dataset(dataset):
    """
    Split into two ~equal-sized parts
    """
    data = dataset.images

    training_data = data[0::2]
    test_data = data[1::2]

    training_target = dataset.target[0::2]
    test_target = dataset.target[1::2]

    training_images = dataset.images[0::2]
    test_images = dataset.images[1::2]

    training = Bunch(data=training_data, target=training_target, images=training_images)
    test = Bunch(data=test_data, target=test_target, images=test_images)

    return training, test


def visualize(rows):
    for r, (title, images) in enumerate(rows):
        for index, (image, label) in enumerate(images):
            plt.subplot(len(rows), len(images), index + 1 + r * len(images))
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            if title:
                plt.title('{}: {}'.format(title, int_to_letter(label)))
            else:
                plt.title(int_to_letter(label))

    plt.show()


def train_and_test_classifier(dataset, classifier, preprocessing_func=None, visualize_n=0):
    training, test = split_dataset(dataset)

    # pca = RandomizedPCA(n_components=1)
    # std_scaler = StandardScaler()
    #
    # X_train = pca.fit_transform(training.data)
    # X_test  = pca.fit_transform(test.data)
    #
    # X_train = std_scaler.fit_transform(X_train)
    # X_test = std_scaler.fit_transform(X_test)
    #
    # classifier = KNeighborsClassifier(n_neighbors=1)
    # classifier.fit(X_train, training.target)

    if preprocessing_func:
        training_data = preprocessing_func(training.data)
        test_data = preprocessing_func(test.data)
    else:
        training_data = training.data
        test_data = test.data

    classifier.fit(training_data, training.target)

    predicted = classifier.predict(test_data)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(test.target, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test.target, predicted))

    if visualize_n:
        images_and_labels = tuple(zip(training.images, training.target))
        images_and_predictions = tuple(zip(test.images, predicted))

        visualize((
            ('', tuple(choice(images_and_labels) for _ in range(visualize_n))),
            ('', tuple(choice(images_and_predictions) for _ in range(visualize_n)))
        ))

    return classifier


if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan, linewidth=np.nan)

    ALPHABET_SIZE = 25
    dataset = load_dataset(n=ALPHABET_SIZE)

    classifier = svm.LinearSVC()

    train_and_test_classifier(
        dataset,
        classifier=classifier,
        preprocessing_func=oriented_gradients,
        visualize_n=20
    )
