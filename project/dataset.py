import csv
import os

import numpy as np
from PIL import Image
from matplotlib.cbook import Bunch

from utils import letter_to_int

PROJECT_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(PROJECT_DIR, 'img')
DATA_DIR = os.path.join(PROJECT_DIR, 'chars74k-lite')
DATA_SHAPE = (-1, 20, 20)
CSV_DIR = os.path.join(PROJECT_DIR, 'csv')


def read_img_to_array(path):
    """
    Gets image as uint8 data
    """
    with Image.open(path).convert('L') as img:
        return tuple(img.getdata())


def read_dataset_from_images(n=25):
    """
    Creates and intermediate representation of the dataset,
     with each image matrix sorted into categories based on the letter it represents
    """
    dataset = {}
    for path, dirs, files in os.walk(DATA_DIR):
        letter = os.path.split(path)[-1]
        files = tuple(filter(lambda file: '.jpg' in file, files))

        if not files:
            continue

        if letter_to_int(letter) > n:
            continue

        files = map(lambda file: os.path.join(path, file), files)

        dataset[letter] = tuple(map(read_img_to_array, files))

    return dataset


def dataset_to_csv(dataset_from_images, write_path='dataset.csv'):
    """
    Takes the intermediate dataset and writes it as a matrix to a csv file.
      When reusing the dataset we only need to read one file directly into a matrix,
      not hundreds of image files.
    """
    with open(write_path, 'w') as outfile:
        writer = csv.writer(outfile)

        for letter, images in dataset_from_images.items():
            letter_int = letter_to_int(letter)
            for img in images:
                writer.writerow(img + (letter_int,))


def get_dataset(n=25):
    """
    Gets dataset from csv if it exists, if not creates it
    """
    path = os.path.join(CSV_DIR, 'dataset_{}.csv'.format(n))
    try:
        return np.loadtxt(path, delimiter=',')
    except FileNotFoundError:
        dataset_from_images = read_dataset_from_images(n)
        dataset_to_csv(dataset_from_images, write_path=path)

        return np.loadtxt(path, delimiter=',')


def load_dataset(n=25):
    """
    Load the dataset from a single big matrix into the Bunch format used by scikit-learn
    """
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
    """
    Splits the dataset (in Bunch format) into two parts
    """
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
