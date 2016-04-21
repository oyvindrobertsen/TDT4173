from functools import reduce

import numpy as np
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler


def binarize(images):
    """
    Thresholding with a dynamically picked threshold value
    """
    images = preprocessing.scale(images)

    binarizer = preprocessing.Binarizer(copy=False).fit(images)
    binarizer.transform(images)

    return images


def oriented_gradients(images, shape=(20, 20)):
    """
    Uses scikit-learn histogram of oriented gradients to find features in the input image.
      By specifying the shape of the input we can adjust the dimensions of the cells used by
       the hog algorithm, ensuring we get a fixed number of features to send to the learner.
       This way we can use images of different sizes with the same learner.
    """
    a, b = shape
    images = images.reshape((-1, a, b))

    cell_dims = [a / x for x in (5, 4, 2)]

    def h(dim, image):
        return hog(
            image,
            orientations=4,
            pixels_per_cell=(dim, dim),
            cells_per_block=(1, 1),
            visualise=False,
            feature_vector=True
        )

    def pool(image):
        f = lambda accumulator, x: np.concatenate((accumulator, h(x, image)))
        return reduce(f, cell_dims, [])

    return np.array([pool(image) for image in images])


def rpca(images):
    pca = RandomizedPCA(n_components=100)
    std_scaler = StandardScaler()

    X_train = pca.fit_transform(images)
    X_train = std_scaler.fit_transform(X_train)

    return X_train
