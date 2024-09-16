from utility import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def build_pdf(x_values):
    n = len(x_values)
    delta = (max(x_values) - min(x_values)) / 2
    print(delta)
    result = [
        list(
            filter(lambda val: elem - delta <= val and val <= elem + delta, x_values)
        ).count(True)
        for elem in x_values
    ]
    # print(result)
    return result


random.seed()

n = 1000
m = 3
x_values = []

for i in range(n):
    x_values.append(max([random.random() for _ in range(m)]))

y_values = build_pdf(x_values)


plt.figure(figsize=(8, 6))

# y_values = [1 / (x_values[i + 1] - x_values[i]) for i in range(n - 1)]
# x_values.pop()

# plt.hist(x_values, bins=n // 100, edgecolor="black")
# sns.kdeplot(
#     x_values,
#     label=f"Underlying Distribution",
#     color="blue",
# )
# plt.scatter(x_values, np.zeros_like(x_values), label="Samples", color="red")

plt.plot(x_values, y_values, marker=".", linestyle="-")

plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Approximating Underlying Distribution from Samples")

# plt.legend()

plt.grid(True)
plt.show()
