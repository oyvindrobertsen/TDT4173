from random import choice

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.datasets.base import Bunch

from project.read_dataset import DATA_SHAPE, get_dataset
from project.utils import int_to_letter

"""
Modified from http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py
"""

N = 1


def load_dataset():
    # each row of the matrix is 400 pixel intensity values and 1 value representing the class of the image
    dataset_matrix = get_dataset(n=N)

    targets = dataset_matrix[:, -1]  # gets the last column, classes
    image_data = dataset_matrix[:, :-1]  # gets every column but the last, image data

    images = image_data.view()
    images.shape = DATA_SHAPE

    return Bunch(data=image_data,
                 targets=targets.astype(np.int),
                 target_names=np.arange(N),
                 images=images,
                 DESCR='')


def split_dataset(dataset):
    """
    Split into two ~equal-sized parts
    """
    data = dataset.images.reshape((len(dataset.images), -1))
    from random import shuffle
    shuffle(data)

    training_data = data[0::2]
    test_data = data[1::2]

    training_targets = dataset.targets[0::2]
    test_targets = dataset.targets[1::2]

    training_images = dataset.images[0::2]
    test_images = dataset.images[1::2]

    training = Bunch(data=training_data, targets=training_targets, images=training_images)
    test = Bunch(data=test_data, targets=test_targets, images=test_images)

    return training, test


def visualize(rows):
    for r, (title, images) in enumerate(rows):
        for index, (image, label) in enumerate(images):
            plt.subplot(len(rows), len(images), index + 1 + r * len(images))
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('{}: {}'.format(title, int_to_letter(label)))

    plt.show()


dataset = load_dataset()
training, test = split_dataset(dataset)

classifier = svm.SVC(gamma=0.001)
classifier.fit(training.data, training.targets)
predicted = classifier.predict(test.data)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test.targets, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test.targets, predicted))

images_and_labels = tuple(zip(training.images, training.targets))
images_and_predictions = tuple(zip(test.images, predicted))

n = 4
visualize((
    ('Training', tuple(choice(images_and_labels) for _ in range(n))),
    ('Prediction', tuple(choice(images_and_predictions) for _ in range(n)))
))
