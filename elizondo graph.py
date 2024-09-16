from utility import *
import matplotlib.pyplot as plt
import os

points = []  # input size, time

file_size = int(input("Gimme image size: "))
directory_path = f"./All images"
all_images = os.listdir(directory_path)
all_images = [
    filename for filename in all_images if int(filename.split("_")[0]) == file_size
]
for filename in all_images:
    print(filename)
    command = [f"./elizondo approximate.exe", filename, "1"]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        approx = float(result.stdout)
    except:
        continue
    actual = float(filename.split("_")[1])
    # points[float(filename.split("_")[1])] = approx
    points.append((approx - actual, approx))
points.sort()
x_points = [p[0] for p in points]
y_points = [p[1] for p in points]

plt.scatter(x_points, y_points, color="blue", marker="o", label="Data Points")
plt.plot(x_points, y_points, color="red", label="Connecting Line")
plt.title("Graph")
plt.xlabel("Actual distance")
plt.ylabel("Approximate distance")
plt.legend()
plt.show()
