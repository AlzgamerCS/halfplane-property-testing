from utility import *
import matplotlib.pyplot as plt
import os

points = {}  # input size, time

tests = int(input("Number of tests: "))
algo_tests = int(input("Number of tests for a single run: "))

directory_path = f"./All images"
all_images = os.listdir(directory_path)
for i in range(tests):
    filename = random.choice(all_images)
    command = [f"./elizondo.exe", filename, str(algo_tests)]
    result = subprocess.run(command, capture_output=True, text=True)
    print(filename)
    n = int(filename.split("_")[0])
    n = n * n
    try:
        time = float(result.stdout)
    except:
        continue
    points.setdefault(n, [])
    points[n].append(time)
x_points = sorted(points.keys())
y_points = [sum(points[n]) / len(points[n]) for n in x_points]

plt.scatter(x_points, y_points, color="blue", marker="o", label="Data Points")
plt.plot(x_points, y_points, color="red", label="Connecting Line")
plt.title("Time-complexity of Elizondo")
plt.xlabel("Input size")
plt.ylabel("Time")
plt.legend()
plt.show()
