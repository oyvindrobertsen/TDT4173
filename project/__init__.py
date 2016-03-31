import os
import re

from PIL import Image

DATA_DIR = 'chars74k-lite'


def read_img_to_array(path):
    with Image.open(path) as img:
        assert img.mode == 'L'  # only dealing with BW images

        return tuple(img.getdata())


def read_dataset(read_f=read_img_to_array, filter_f=None):
    dataset = {}
    for path, dirs, files in os.walk(DATA_DIR):
        letter = os.path.split(path)[-1]
        files = tuple(filter(lambda file: '.jpg' in file, files))

        if not files:
            continue

        if filter_f:
            files = filter(filter_f, files)

        files = map(lambda file: os.path.join(path, file), files)

        dataset[letter] = tuple(map(read_f, files))

    return dataset


even = lambda file_name: int(re.findall(r'\d+', file_name)[0]) % 2 == 0
odd = lambda file_name: not even(file_name)
print(*read_dataset(filter_f=even).items(), sep='\n')
