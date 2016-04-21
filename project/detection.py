import numpy as np

from dataset import read_img_to_array
from utils import get_image_shape


def sliding_windows(path, stride, window_size):
    window_w, window_h = window_size

    shape = img_w, img_h = get_image_shape(path)
    is_within_bounds = lambda l, r, t, b: (l >= 0 and t >= 0) and (r <= img_w and b <= img_h)

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

            row = np.append(window.reshape(window_w * window_h), (x, y))

            m.append(row)

    return np.array(m)
