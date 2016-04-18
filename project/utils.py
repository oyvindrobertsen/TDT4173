from PIL import Image


def letter_to_int(letter):
    return ord(letter) - ord('a')


def int_to_letter(n):
    return chr(ord('a') + int(n))


def get_image_shape(path):
    with Image.open(path) as img:
        return img.size
