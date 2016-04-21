from random import choice

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from dataset import split_dataset
from utils import int_to_letter


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
