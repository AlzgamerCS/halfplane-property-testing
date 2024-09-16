from utility import *
import sys


def main():
    filename = input("Give me a filename: ")
    epsilon = eval(input("Epsilon (sample): "))
    delta = eval(input("Delta: "))
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
    exe = "./approximate line.exe"
    command = [exe, filename, epsilon, delta]
    command = [str(val) for val in command]
    result = subprocess.run(command, capture_output=True, text=True)
    print(str(result.stdout))
    print(str(result.stderr))
    wb = result.stdout.split()
    if exe == "./approximate line svm.exe":
        wb = wb[-3:]
    else:
        print(wb[-1])
        wb.pop()
    wb = [float(val) for val in wb]
    print(wb)
    dot_format = True
    image = matrix_to_image(matrix, pixel_size=2, wb=wb[:3], dot_format=dot_format)
    if dot_format:
        plt.show()
    else:
        image.show()
    plt.show()
    if input("Save (y/n)?: ") == "y":
        if dot_format:
            plt.savefig(f"Images png/{filename}.png")
        else:
            image.save(f"Images png/{filename}.png")

    # wb[-1] *= -1
    # print(wb)
    # image = matrix_to_image(matrix, pixel_size=2, wb=wb[:3])
    # image.show()
    # if input("Save (y/n)?: ") == "y":
    #     image.save(f"Images png/{filename}.png")


if __name__ == "__main__":
    main()
