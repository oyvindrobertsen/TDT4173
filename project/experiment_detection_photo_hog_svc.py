import os
import matplotlib.pyplot as plt
from sklearn.svm.classes import LinearSVC
from dataset import load_dataset, IMG_DIR
from detection import sliding_windows
from preprocessing import oriented_gradients, binarize
from experiment_training import train_and_test_classifier
from utils import int_to_letter

path = os.path.join(IMG_DIR, "illuminati.jpg")
window_size = a, b = (50, 50)
stride = 10
threshold = 0.5
make_pre_func = lambda shape: (lambda images: oriented_gradients((images), shape=shape))
classifier = LinearSVC()

if __name__ == '__main__':
    classifier = train_and_test_classifier(
        load_dataset(),
        classifier=classifier,
        preprocessing_func=make_pre_func(shape=(20, 20))
    )

    print("\nFinding windows\n")

    windows = sliding_windows(path, stride, window_size)

    target = windows[:, -2:]  # xy columns
    image_data = windows[:, :-2]  # image data

    preprocess_test = make_pre_func(shape=window_size)
    test_data = preprocess_test(image_data)

    print("\nTesting windows\n")

    decisions = classifier.decision_function(test_data)

    z = zip(decisions, target)
    z = sorted(z, key=lambda dt: -max(dt[0]))

    print(*[max(d) for d, t in z], sep='\n')

    im = plt.imread(path)
    implot = plt.imshow(im, cmap=plt.cm.gray)

    for d, t in z:
        m = max(d)
        letter = int_to_letter(max(range(25), key=lambda n: d[n]))
        x, y = t
        if m > threshold:
            plt.annotate(letter, (x, y), color='r', size=20)

    plt.show()
