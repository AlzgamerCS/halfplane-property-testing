from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import sys
from utility import *

filename = input("Give me a filename: ")
n = int(filename.split("_")[0])
matrix = [None for _ in range(n)]

with open("./All images/" + filename, "r") as file:
    original_stdin = sys.stdin
    sys.stdin = file
    n = int(input())
    for i in range(n):
        line = input()
        matrix[i] = line
    sys.stdin = original_stdin

image = matrix_to_image(matrix, pixel_size=int(1000 / n))
image.show()


points = []
labels = []
for i in range(n):
    for j in range(n):
        points.append((i, j))
        if matrix[i][j] == black_pixel:
            labels.append(1)
        else:
            labels.append(0)

X = np.array(points)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm = SVC(kernel="linear", C=0.5, random_state=42)
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))


w = svm.coef_[0]
b = svm.intercept_[0]
print(
    f"The equation of the separation line is: {w[0]:.2f}*x0 + {w[1]:.2f}*x1 + {b:.2f} = 0"
)


plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors="k")


ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()


xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)


ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)

# Highlight the support vectors
# ax.scatter(
#     svm.support_vectors_[:, 0],
#     svm.support_vectors_[:, 1],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Linear Classification with Decision Boundary")
plt.show()
