import numpy as np
from skimage.transform._warps import resize
from sklearn.datasets.base import load_digits
from sklearn.svm.classes import LinearSVC

from read_dataset import read_img_to_array
from utils import get_image_shape


def sliding_windows(path, stride, window_size):
    window_w, window_h = window_size

    shape = img_w, img_h = get_image_shape(path)

    arr = np.array(read_img_to_array(path), dtype=np.uint8, order='C')
    arr = np.reshape(arr, (img_h, img_w))

    windows = []

    dw = window_w // 2
    dh = window_h // 2
    for y in range(dh, img_h, stride):
        for x in range(dw, img_w, stride):
            x_a = x - dw
            x_b = x + dw
            y_a = y - dh
            y_b = y + dh

            if x_a < 0 or x_b > img_w or y_a < 0 or y_b > img_h:
                continue

            window = arr[y_a:y_b, x_a:x_b]

            windows.append(window)

    return np.array(windows)


def resize_imgs(imgs, size):
    return np.array(tuple(map(lambda img: resize(img, size), imgs)))


def flatten_imgs(imgs):
    n_imgs = len(imgs)
    w, h = imgs[0].shape
    n_px = w * h
    return imgs.reshape((n_imgs, n_px))


if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan, linewidth=np.nan)

    dataset = load_digits()

    classifier = LinearSVC()
    classifier.fit(dataset.data, dataset.target)

    path = "OCR-font-samples.jpg"

    windows = sliding_windows(
        path,
        stride=5,
        window_size=(15, 15)
    )

    print(windows[40:60])

    # windows = resize_imgs(windows, (8, 8))

    # print(classifier.predict(flatten_imgs(windows)))
