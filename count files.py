from utility import *
import os

root_dir = "."

folder_path = root_dir + "/All images"
print("Started")
table = {}
for filename in os.listdir(folder_path):
    name_parts = filename.split("_")
    n, m = int(name_parts[0]), float(name_parts[1])
    table[n] = table.get(n, 0) + 1
for size in sorted(table.keys()):
    print(f"Size {size}x{size}: {table[size]} images")
print(f"Total: {sum(table.values())} images")
