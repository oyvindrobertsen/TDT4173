def letter_to_int(letter):
    return ord(letter) - ord('a')


def int_to_letter(n):
    return chr(ord('a') + int(n))
