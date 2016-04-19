from sklearn import preprocessing
import numpy as np

from skimage.feature import hog
from skimage import data, color, exposure

def binarize(images):
    images = preprocessing.scale(images)

    binarizer = preprocessing.Binarizer(copy=False).fit(images)
    binarizer.transform(images)

    return images

def oriented_gradients(images, celldim=(5, 5), imageshape=(20,20)):

    images = images.reshape((len(images), imageshape[0], imageshape[1]))

    def h(image):
        return hog(image, orientations = 4, pixels_per_cell = (celldim[0], celldim[1]),  
                cells_per_block = (1,1), visualise=False, feature_vector=True)

    return np.array([h(image) for image in images])

