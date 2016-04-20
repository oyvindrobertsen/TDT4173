from random import choice

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.datasets.base import Bunch

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from preprocessing import binarize, oriented_gradients, rpca
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
                 target_names=np.arange(n),
                 images=images,
                 DESCR='')


def split_dataset(dataset, split=0.8):
    training_data = []
    test_data = []

    training_target = []
    test_target = []

    training_images = []
    test_images = []

    c = 0
    for data, target, image in zip(dataset.data, dataset.target, dataset.images):
        if c / 10 < split:
            training_data.append(data)
            training_target.append(target)
            training_images.append(image)
        else:
            test_data.append(data)
            test_target.append(target)
            test_images.append(image)

        c = (c + 1) % 10

    training = Bunch(
        data=np.array(training_data),
        target=np.array(training_target),
        images=np.array(training_images)
    )
    test = Bunch(
        data=np.array(test_data),
        target=np.array(test_target),
        images=np.array(test_images)
    )

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
    np.set_printoptions(threshold=np.nan, linewidth=np.nan)
    training, test = split_dataset(dataset)

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
    ALPHABET_SIZE = 25
    dataset = load_dataset(n=ALPHABET_SIZE)

    classifier = svm.LinearSVC()
    # classifier = KNeighborsClassifier(
    #         n_neighbors=20,
    #         algorithm='kd_tree',
    #         weights='distance'
    #         )

    combined = lambda images: oriented_gradients(binarize(images))

    # classifier.fit(combined)

    train_and_test_classifier(
        dataset,
        classifier=classifier,
        preprocessing_func=combined,
        visualize_n=20
    )
