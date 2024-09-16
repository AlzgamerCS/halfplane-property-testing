from utility import *
import sys


def main():
    filename = input("Give me a filename: ")
    # tests = int(input("Number of tests: "))
    n = int(filename.split("_")[0])
    matrix = [None for _ in range(n)]

    with open("./All images/" + filename, "r") as file:
        original_stdin = sys.stdin
        sys.stdin = file
        n = int(input())
        for i in range(n):
            line = input()
            matrix[i] = line
        sys.stdin = original_stdin
    image = matrix_to_image(matrix)
    image.show()
    if input("Save (y/n)?: ") == "y":
        image.save(f"Images png/{filename}.png")


if __name__ == "__main__":
    main()
