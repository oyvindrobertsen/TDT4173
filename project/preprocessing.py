from sklearn import preprocessing
import numpy as np

from skimage.feature import hog
from skimage import data, color, exposure

from functools import reduce

def binarize(images):
    images = images.reshape((len(images), 400))
    images = preprocessing.scale(images)

    binarizer = preprocessing.Binarizer(copy=False).fit(images)
    binarizer.transform(images)

    return images

def oriented_gradients(images, imageshape=(20,20)):

    images = images.reshape((len(images), imageshape[0], imageshape[1]))

    celldims = [imageshape[0]/x for x in [8.0, 4.0, 2.0]]

    def h(dim, image):
        return hog(image, orientations = 4, pixels_per_cell = (dim, dim),  
                cells_per_block = (1,1), visualise=False, feature_vector=True)

    def pool(image):
        return reduce(lambda accumulator, x: np.concatenate((accumulator, h(x, image))), celldims, [])

    return np.array([pool(image) for image in images])

