import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("table.csv")
print(df)
print(len(df))

row_numbers = range(len(df))


plt.figure(figsize=(10, 6))


plt.plot(row_numbers, df["Elizondo-Time"], "r", label="Elizondo Time")


plt.plot(row_numbers, df["Tester-Time"], "b", label="Tester Time")

plt.xlabel("File Number")
plt.ylabel("Time")
plt.title("Elizondo vs Tester")
plt.legend()


plt.savefig("./plot.png")
plt.show()
