from sklearn import svm

from dataset import load_dataset
from preprocessing import binarize, oriented_gradients
from train import train_and_test_classifier

if __name__ == '__main__':
    dataset = load_dataset()

    classifier = svm.LinearSVC()

    combined = lambda images: oriented_gradients(binarize(images))

    train_and_test_classifier(
        dataset,
        classifier=classifier,
        preprocessing_func=combined,
        visualize_n=5
    )
