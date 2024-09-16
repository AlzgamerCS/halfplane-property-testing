from utility import *
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import seaborn as sns
import subprocess
from statistics import pstdev

table = []
filename = input("Give me a filename: ")
err = eval(input("Epsilon (sample): "))
shift = eval(input("Shift: "))
tests = int(input("Number of tests: "))

for i in range(tests):
    command = ["./approximate.exe", filename, err, shift]
    command = [str(val) for val in command]
    result = subprocess.run(command, capture_output=True, text=True)
    dist = float(result.stdout)
    table.append(dist)
print(f"Average: {sum(table) / len(table)}")
print(f"Standard deviation: {pstdev(table)}")
x_values = table

plt.figure(figsize=(16, 9))
sns.kdeplot(
    x_values,
    label=f"Underlying Distribution (Actual distance: {filename.split('_')[1]})",
    color="blue",
)

plt.scatter(x_values, np.zeros_like(x_values), label="Samples", color="red")

plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Approximating Underlying Distribution from Samples")

plt.legend()

plt.grid(True)
plt.show()
