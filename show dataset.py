from utility import *
import sys


def main():
    filename = input("Give me a filename: ")
    # tests = int(input("Number of tests: "))
    n = int(filename[1:].split("_")[0])

    points = []
    with open(f"{datasets_path}/{filename}", "r") as file:
        original_stdin = sys.stdin
        sys.stdin = file
        n = int(input())
        for i in range(n):
            line = input()
            line = line.split()
            points.append([float(line[0]), float(line[1]), line[2]])
        sys.stdin = original_stdin
    A = [p for p in points if p[2] == black_pixel]
    B = [p for p in points if p[2] == white_pixel]
    x, y = [p[0] for p in A], [p[1] for p in A]
    plt.scatter(x, y, color="red", marker=".")
    x, y = [p[0] for p in B], [p[1] for p in B]
    plt.scatter(x, y, color="blue", marker=".")
    plt.show()


if __name__ == "__main__":
    main()
