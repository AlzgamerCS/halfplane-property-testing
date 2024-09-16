from utility import *
import os

n = int(input("Image size: "))
eps = float(input("Epsilon distance: "))

root = "."
filelist = os.listdir(f"{root}/All images")

filelist = [file for file in filelist if int(file.split("_")[0]) == n]

filelist.sort(key=lambda file: float(file.split("_")[1]))

target_file = None
for file in filelist:
    if float(file.split("_")[1]) <= eps:
        target_file = file
    else:
        break

with open(f"All images/{target_file}") as file:
    lines = file.readlines()[1:]
    matrix = [list(line.strip()) for line in lines]
    img = matrix_to_image(matrix)
    print(float(target_file.split("_")[1]))
    img.show()
    if input("Save (y/n)?: ") == 'y':
        img.save(f"Images png/{target_file}.png")
