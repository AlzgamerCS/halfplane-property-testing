import os
from utility import *
from shutil import copyfile


def split_str(s):
    strs = s[1:].split("_")
    return [int(strs[0]), float(strs[1]), strs[-1]]


folder_path = "./Datasets"
files = os.listdir(folder_path)
files = sorted(files, key=split_str)

for idx, file in enumerate(files):
    source = f"{folder_path}/{file}"
    copyfile(source, "./input.txt")
    n = split_str(file)[0]
    ans = int(split_str(file)[1] * n)

    command = (
        "powershell -Command \"Get-Content .\\input.txt | & '.\\epsilon distance.exe'\""
    )
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    ans1 = result.stderr.strip()
    ans1 = result.stdout.strip()

    command = 'powershell -Command "Get-Content .\\input.txt | .\\n2logn.exe"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    ans2 = result.stderr.strip()
    ans2 = result.stdout.strip()

    ans1 = int(ans1)
    ans2 = int(ans2)
    if len(set([ans, ans1, ans2])) != 1:
        print(f"{idx} - {file}: {ans}, {ans1}, {ans2}")

print("Finished!")
