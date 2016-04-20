import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn.svm.classes import LinearSVC

from main import load_dataset, train_and_test_classifier
from preprocessing import oriented_gradients
from read_dataset import read_img_to_array
from utils import get_image_shape, int_to_letter


def sliding_windows(path, stride, window_size):
    window_w, window_h = window_size

    shape = img_w, img_h = get_image_shape(path)
    is_within_bounds = lambda l, r, t, b: (l >= 0 and t >= 0) and (r < img_w and b < img_h)

    arr = np.array(read_img_to_array(path), dtype=np.uint8)
    arr = np.reshape(arr, (img_h, img_w))

    m = []

    dw = window_w // 2
    dh = window_h // 2
    for y in range(dh, img_h, stride):
        for x in range(dw, img_w, stride):
            left = x - dw
            right = x + dw
            top = y - dh
            bottom = y + dh

            if not is_within_bounds(left, right, top, bottom):
                continue

            window = arr[top:bottom, left:right]

            row = np.append(window.reshape(a * b), (x, y))

            m.append(row)

    return np.array(m)


def resize_imgs(imgs, size):
    return np.array(tuple(map(lambda img: resize(img, size), imgs)))


def flatten_imgs(imgs):
    n_imgs = len(imgs)
    w, h = imgs[0].shape
    n_px = w * h
    return imgs.reshape((n_imgs, n_px))


if __name__ == '__main__':
    dataset = load_dataset()

    classifier = train_and_test_classifier(
        dataset,
        classifier=LinearSVC(),
        preprocessing_func=lambda images: oriented_gradients((images), shape=(20, 20))
    )

    print("\nFinding windows\n")

    path = "doodeedoo.jpg"

    window_size = a, b = (50, 50)
    windows = sliding_windows(
        path,
        stride=10,
        window_size=window_size
    )

    target = windows[:, -2:]  # xy columns
    image_data = windows[:, :-2]  # image data

    preprocess_test = lambda images: oriented_gradients((images), shape=window_size)
    test_data = preprocess_test(image_data)

    print("\nTesting windows\n")

    decisions = classifier.decision_function(test_data)

    z = zip(decisions, target)
    z = sorted(z, key=lambda dt: -max(dt[0]))

    max_vals = [max(d) for d, t in z[:10]]

    im = plt.imread(path)
    implot = plt.imshow(im, cmap=plt.cm.gray)

    for d, t in z:
        m = max(d)
        letter = int_to_letter(max(range(25), key=lambda n: d[n]))
        if m > 1.0:
            a, b = t
            plt.annotate(letter, (a, b), color='r', size=int(10 * m))

    plt.show()
