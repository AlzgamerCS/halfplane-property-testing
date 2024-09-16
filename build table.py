from utility import *
from bisect import bisect_right
from math import ceil
import pandas as pd
import os


def conv(filename):
    parts = filename.split("_")
    return int(parts[0]), float(parts[1])


tests = 1000
images_path = "./All images"

filenames = sorted(os.listdir(images_path), key=conv)
filenames = [filename for filename in filenames if conv(filename)[1] != 0]


sizes = sorted(set([conv(filename)[0] for filename in filenames]))
epsvals = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]

sample_files = []

for size in sizes:
    for eps in epsvals:
        search_key = (size, eps)
        i = bisect_right(filenames, search_key, key=conv)
        if i >= len(filenames):
            i = len(filenames) - 1
        closest_file = filenames[random.choice([i - 1, i])]
        sample_files.append(closest_file)

filenames = sorted(set(sample_files), key=conv)

print(filenames)

data = {
    "Size": [],
    "Epsilon": [],
    "Elizondo_Accuracy": [],
    "Elizondo_Time": [],
    "Sample": [],
    "Tester_Accuracy": [],
    "Tester_Time": [],
}

i = 1
for filename in filenames:
    curr_size = len(data["Size"])
    n, epsilon = conv(filename)
    try:
        command = ["./elizondo.exe", str(tests), filename]
        results = subprocess.run(command, capture_output=True, text=True).stdout
        results = results.split()
        accuracy, time = float(results[0]), float(results[1])
        data["Size"].append(n)
        data["Epsilon"].append(round(epsilon, 6))
        data["Elizondo_Accuracy"].append(round(accuracy * 100, 2))
        data["Elizondo_Time"].append(round(1000 * time, 6))

        epsilon = min(epsilon, 1 / 6) / 8
        if epsilon == 0:
            epsilon = 1 / (n**2)

        command = ["./tester.exe", str(tests), filename, str(epsilon)]
        results = subprocess.run(command, capture_output=True, text=True).stdout
        results = results.split()
        accuracy, time = float(results[0]), float(results[1])
        data["Sample"].append(int(ceil(1 / epsilon)))
        data["Tester_Accuracy"].append(round(accuracy * 100, 2))
        data["Tester_Time"].append(round(1000 * time, 6))
    except:
        print(filename)
    strip = False
    for k, v in data.items():
        if len(v) == curr_size:
            strip = True
            break
    if strip:
        for k, v in data.items():
            if len(v) != curr_size:
                v.pop()
    print(f"\rProgress: {round(i / len(filenames) * 100, 2)}%  ", end="", flush=True)
    i += 1

df = pd.DataFrame(data)
df.to_csv("./table.csv", index=False)
print("Data saved into table.csv")
