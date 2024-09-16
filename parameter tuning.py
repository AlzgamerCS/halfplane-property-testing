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

seed(44)
n = int(argv[1])
to_change = 0.4
C_param = 0.00001

graph = []

for _ in range(100):
    points = generate_halfplane(n)
    change_labels(points, int(n * to_change))

    X = np.array([(x, y) for x, y, label in points])
    y = np.array([(-1 if label == black_pixel else 1) for x, y, label in points])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    C_param = random() * 10
    svm = SVC(kernel="linear", C=C_param, random_state=42)

    classifier = svm
    classifier.fit(X_train, y_train)
    w = classifier.coef_[0]
    b = svm.intercept_[0]
    # print(
    #     f"The equation of the separation line is: {w[0]}*x0 + {w[1]}*x1 + {b} = 0. Slope = {-w[1] / w[0]}"
    # )
    y_pred = classifier.predict(X_test)
    accuracy = sum([int(y_pred[i] == y_test[i]) for i in range(len(y_test))]) / len(
        y_test
    )
    graph.append([C_param, accuracy])
    print(graph[-1])
    C_param *= 2


plt.figure(figsize=(16, 9))
graph = np.array(graph)
plt.scatter(x=graph[:, 0], y=graph[:, 1], color="red")
plt.xlabel("C - parameter")
plt.ylabel("Average accuracy")
plt.ylim(0, 1)
plt.title("Graph")
plt.show()
# print(classification_report(y_test, y_pred))
# report = classification_report(y_test, y_pred, output_dict=True)

"""
plt.figure(figsize=(8, 6))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap=plt.cm.Paired,
    edgecolors="k",
    label="Training Data",
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap=plt.cm.Paired,
    edgecolors="k",
    alpha=0.5,
    label="Test Data",
)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = classifier.decision_function(xy).reshape(XX.shape)


ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Linear Classification with Decision Boundary")
plt.legend()
plt.show()
plt.close()
"""
# exit()
