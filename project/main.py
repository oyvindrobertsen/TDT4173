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

from sklearn.feature_extraction.image import img_to_graph


"""
Modified from http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py
"""

N = 25
np.set_printoptions(threshold=np.nan, linewidth=np.nan)

def load_dataset(preprocessing_func=None):
    # each row of the matrix is 400 pixel intensity values and 1 value representing the class of the image
    dataset_matrix = get_dataset(n=N)

    target = dataset_matrix[:, -1]  # gets the last column, classes
    image_data = dataset_matrix[:, :-1]  # gets every column but the last, image data

    images = image_data.view()

    if preprocessing_func:
        images = preprocessing_func(images)

    # images.shape = DATA_SHAPE

    return Bunch(data=image_data,
                 target=target.astype(np.int),
                 target_names=np.arange(N),
                 images=images,
                 DESCR='')


def split_dataset(dataset):
    """
    Split into two ~equal-sized parts
    """
    # data = dataset.images.reshape((len(dataset.images), -1))
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
            plt.title('{}: {}'.format(title, int_to_letter(label)))

    plt.show()


dataset = load_dataset(preprocessing_func=oriented_gradients)
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

classifier = svm.LinearSVC()
classifier.fit(training.data, training.target)

predicted = classifier.predict(test.data)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test.target, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test.target, predicted))

images_and_labels = tuple(zip(training.images, training.target))
images_and_predictions = tuple(zip(test.images, predicted))


n = 16
visualize((
    ('', tuple(choice(images_and_labels) for _ in range(n))),
    ('', tuple(choice(images_and_predictions) for _ in range(n)))
))
