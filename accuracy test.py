from utility import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import sys
from random import seed
from sys import argv

seed(10)

results = []
for n in range(10, 1000):
    points = generate_halfplane(n)

    X = np.array([(x, y) for x, y, label in points])
    y = np.array([(0 if label == black_pixel else 1) for x, y, label in points])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    svm = SVC(kernel="linear", C=100, random_state=42)
    classifier = svm
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    svm_acc = report["accuracy"]

    approximator = Approximator(epsilon=1 / len(X_test), delta=0.001)
    classifier = approximator
    classifier.fit(X_train, y_train)


    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    appr_acc = report["accuracy"]

    results.append((svm_acc, appr_acc))

plt.scatter(range(10, 1000), )
filename = input("Give me a filename: ")
epsilon = eval(input("Epsilon (sample): "))
tests = int(input("Number of tests: "))
n = int(filename.split("_")[0])

filename = f"./{"All images" if filename[0] == 'D' else "Datasets"}/{filename}"

with open(filename, "r") as file:
    original_stdin = sys.stdin
    sys.stdin = file
    n = int(input())
    for i in range(n):
        line = input()
        matrix[i] = line
    sys.stdin = original_stdin
