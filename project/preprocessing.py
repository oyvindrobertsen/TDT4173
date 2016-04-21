from sklearn import preprocessing
import numpy as np

from skimage.feature import hog
from skimage import data, color, exposure

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler

from functools import reduce

def binarize(images):
    images = images.reshape((len(images), 400))
    images = preprocessing.scale(images)

    binarizer = preprocessing.Binarizer(copy=False).fit(images)
    binarizer.transform(images)

    return images

def oriented_gradients(images, imageshape=(20,20)):

    images = images.reshape((len(images), imageshape[0], imageshape[1]))

    celldims = [imageshape[0]/x for x in [5, 4, 2]]

    def h(dim, image):
        return hog(image, orientations = 4, pixels_per_cell = (dim, dim),  
                cells_per_block = (1,1), visualise=False, feature_vector=True)

    def pool(image):
        return reduce(lambda accumulator, x: np.concatenate((accumulator, h(x, image))), celldims, [])

    return np.array([pool(image) for image in images])

def rpca(images):

    pca = RandomizedPCA(n_components=100)
    std_scaler = StandardScaler()
    
    X_train = pca.fit_transform(images)
    X_train = std_scaler.fit_transform(X_train)

    return X_train
