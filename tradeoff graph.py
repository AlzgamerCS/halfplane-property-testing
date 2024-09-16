from utility import *
import matplotlib.pyplot as plt
import os

file_count = 1
tests = 100
file_list = sorted(
    os.listdir("./All images"),
    key=lambda name: (int(name.split("_")[0]), float(name.split("_")[0])),
)[:300]

file_list = [random.choice(file_list) for _ in range(file_count)]

for filename in file_list:
    print(filename)
    vals = []

    n = int(filename.split("_")[0])
    dist = float(filename.split("_")[1])
    count = 0
    for s in range(1, n**2):
        eps = 1 / s
        command = [f"./tester.exe", str(tests), filename, str(eps)]
        results = subprocess.run(command, capture_output=True, text=True).stdout.split()
        # print(results)
        vals.append((s / n**2, float(results[0])))
        if float(results[0]) >= 0.95:
            count += 1
        if count >= 100:
            break
    vals.sort()
    x_values = [val[0] for val in vals]  # time
    y_values = [val[1] for val in vals]  # accuracy

    plt.plot(x_values, y_values, color="b", linestyle="-", label="Data")

    plt.xlabel("Sample size")
    plt.ylabel("Accuracy")
    plt.title(filename)

    plt.grid(True)
    plt.legend()
    plt.show()
