import csv
import os

import numpy as np
from PIL import Image

from utils import letter_to_int

DATA_DIR = 'chars74k-lite'
DATA_SHAPE = (-1, 20, 20)


def read_img_to_array(path):
    with Image.open(path) as img:
        assert img.mode == 'L'  # only dealing with Monochrome images

        return tuple(img.getdata())


def read_dataset_from_images(n=25):
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
    with open(write_path, 'w') as outfile:
        writer = csv.writer(outfile)

        for letter, images in dataset_from_images.items():
            letter_int = letter_to_int(letter)
            for img in images:
                writer.writerow(img + (letter_int,))


def get_dataset(n=25):
    path = 'dataset_{}.csv'.format(n)
    try:
        return np.loadtxt(path, delimiter=',')
    except FileNotFoundError:
        dataset_from_images = read_dataset_from_images(n)
        dataset_to_csv(dataset_from_images, write_path=path)

        return np.loadtxt(path, delimiter=',')
