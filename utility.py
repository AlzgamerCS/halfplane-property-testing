from PIL import Image, ImageDraw
from typing import List
import random
from random import random, randint, choice, uniform
from time import sleep
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

black_pixel = "#"
white_pixel = "."
datasets_path = "./Datasets"
images_path = "./All images"


def matrix_to_string(matrix: List[List[str]]) -> str:
    return "".join([cell for row in matrix for cell in row])


def matrix_to_image(
    matrix, pixel_size=1, wb=None, dot_format=False
) -> Image.Image:  # wb = weight-bias vector for a line to be drawn
    n = len(matrix)
    if not dot_format:
        black_color = "black"
        white_color = "white"
        image = Image.new("RGB", (n * pixel_size, n * pixel_size))
        draw = ImageDraw.Draw(image)
        for y in range(n):
            for x in range(n):
                color = black_color if matrix[y][x] == black_pixel else white_color
                draw.rectangle(
                    [
                        (x * pixel_size, y * pixel_size),
                        ((x + 1) * pixel_size - 1, (y + 1) * pixel_size - 1),
                    ],
                    fill=color,
                )
        if wb is not None:
            for y in range(n):
                if wb[1] != 0:
                    x = int(round(-(wb[0] * y + wb[2]) / wb[1]))
                else:
                    x = int(round(-(wb[1] * y + wb[2]) / wb[0]))
                    x, y = y, x
                if x < 0 or x >= n or y < 0 or y >= n:
                    continue
                color = "red"
                draw.rectangle(
                    [
                        (x * pixel_size, y * pixel_size),
                        ((x + 1) * pixel_size - 1, (y + 1) * pixel_size - 1),
                    ],
                    fill=color,
                )

        # for y in range(n):
        #         x = n // 2
        #         color = "red"
        #         draw.rectangle(
        #             [
        #                 (x * pixel_size, y * pixel_size),
        #                 ((x + 1) * pixel_size - 1, (y + 1) * pixel_size - 1),
        #             ],
        #             fill=color,
        #         )
        return image
    black_points = []
    white_points = []
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == black_pixel:
                black_points.append((i, j))
            else:
                white_points.append((i, j))
    black_x, black_y = zip(*black_points)
    white_x, white_y = zip(*white_points)
    dot_size = 10
    # Plot the points
    plt.scatter(white_x, white_y, color="yellow", s=dot_size, label="White points")
    plt.scatter(black_x, black_y, color="blue", s=dot_size, label="Black points")

    def line_equation(x):
        return -(wb[0] * x + wb[2]) / wb[1]

    if wb is not None:
        x_line = np.linspace(0, n - 1, n)
        y_line = line_equation(x_line)
        plt.plot(x_line, y_line, color="red", label="Line")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Points and Line (optional) Plot")
    plt.legend()


def build_random(n):
    return build_p_image(n, 0.5)


def get_random_line(n):
    x1 = uniform(-1, n)
    y1 = uniform(-1, n)
    x2 = uniform(-1, n)
    y2 = uniform(-1, n)
    d = x1 * y2 - x2 * y1
    if d == 0:
        return (y1, -x1, 0)
    else:
        return ((y1 - y2) / d, (x2 - x1) / d, 1)


def build_halfplane(n):
    ...
    # return build_function_image(n, lambda x: slope * x + b)


def build_epsilon_close(n, epsilon, max_possible=True):
    max_count = int(n * n * epsilon)
    change_count = max_count if max_possible else randint(1, max_count)
    matrix = build_halfplane(n)
    for _ in range(change_count):
        y = randint(0, n - 1)
        x = randint(0, n - 1)
        matrix[y][x] = not matrix[y][x]
    return matrix


def build_p_image(n, p):
    return [[(True if random() <= p else False) for _ in range(n)] for _ in range(n)]


def build_function_image(n, f):
    matrix = [[None for _ in range(n)] for _ in range(n)]
    for y in range(n):
        for x in range(n):
            color = True if y > f(x) else False
            matrix[y][x] = color
    return matrix


def find_line_distance(matrix, p1, p2, min_distance):
    n = len(matrix)
    vertical = p2[0] == p1[0]
    slope = 0
    b = 0
    if not vertical:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - slope * p1[0]
    black_more = 0
    white_more = 0
    black_less = 0
    white_less = 0
    for y in range(n):
        for x in range(n):
            if (vertical and x <= p1[0]) or y >= slope * x + b:
                if matrix[y][x]:
                    black_more += 1
                else:
                    white_more += 1
            else:
                if matrix[y][x]:
                    black_less += 1
                else:
                    white_less += 1
            if min(white_more + black_less, black_more + white_less) >= min_distance:
                return min_distance
    return min(white_more + black_less, black_more + white_less)


def find_epsilon_distance(matrix):
    n = len(matrix)
    black_count = sum(
        [1 if cell == black_pixel else 0 for row in matrix for cell in row]
    )
    min_distance = min(black_count, n * n - black_count)
    for i in range(n * n):
        for j in range(i):
            p1 = [i % n, i // n]
            p2 = [j % n, j // n]
            min_distance = min(
                min_distance, find_line_distance(matrix, p1, p2, min_distance)
            )
    return min_distance / (n * n)


def find_point_line_distance(p1, p2, point):
    vertical = p2[0] == p1[0]
    m0 = 0
    b0 = 0
    if not vertical:
        m0 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b0 = p1[1] - m0 * p1[0]
    x0, y0 = point
    m1 = -1 / m0
    b1 = y0 - m1 * x0
    x1 = (b1 - b0) / (m0 - m1)
    y1 = m0 * x1 + b0
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


def build_gradient(n, pdf):
    matrix = [[None for _ in range(n)] for _ in range(n)]
    x1 = uniform(-1, n)
    y1 = uniform(-1, n)
    x2 = uniform(-1, n)
    y2 = uniform(-1, n)
    slope = (y2 - y1) / (x2 - x1)
    b = y1 - slope * x1
    for y in range(n):
        for x in range(n):
            dist = find_point_line_distance([x1, y1], [x2, y2], [x, y])
            if y < slope * x + b:
                dist = -dist
            matrix[y][x] = random() <= pdf(dist)
    return matrix


# def get_images(image_size):
#     file_names = os.listdir(images_path)
#     selected_images = []
#     for name in file_names:
#         with open(images_path + name, "r") as file:
#             if len(file.readlines()) == image_size:
#                 selected_images.append(name)
#     return sorted(selected_images)


# def show(epsilon_distance, image_size):
#     print(f"{image_size}x{image_size} image will be shown")
#     labels = get_images(image_size)

#     dist_str = format(epsilon_distance, ".9f")
#     dist_str = labels[bisect.bisect_left(labels, dist_str)] if labels else ""
#     if not dist_str:
#         print("No such image exists")
#         return
#     # file_name = root + dist_str
#     lines = []
#     try:
#         with open(file_name, "r") as file:
#             lines = file.readlines()
#     except:
#         pass
#     image_size = len(lines)
#     matrix = [
#         [(True if line[i] == "#" else False) for i in range(image_size)]
#         for line in lines
#     ]
#     image = matrix_to_image(matrix, 1000 // image_size)
#     image.show()
#     print(f"Epsilon distance = {dist_str}")


def test(): ...


id_process_table = {}


def is_running(pid):
    return id_process_table[pid].poll() is None


def start_process(size):
    process = subprocess.Popen(
        [
            "./generate dataset.exe",
            f"{size}",
            f"{random() / 2}",
        ]
    )
    pid = process.pid
    print(f"Size = {size}, pid = {pid}")
    id_process_table[pid] = process
    return pid


def generate_indefinitely():
    sizes = list(range(100, 10001, 100))

    max_process_count = 12

    index = 0
    process_ids = []
    count = 0
    folder_path = f"/home/pb_ra/Kuanysh_Murat_RA_Project/halfplane-property-testing/Datasets"
    folder_size = get_folder_size(folder_path)
    print(folder_size)
    while folder_size < 2.0:
        new_process_ids = []
        for pid in process_ids:
            if is_running(pid):
                new_process_ids.append(pid)
        process_ids = new_process_ids
        while len(process_ids) < max_process_count:
            pid = start_process(sizes[index])
            process_ids.append(pid)
            index += 1
            if index >= len(sizes):
                count += 1
                index = 0
                folder_size = get_folder_size(folder_path)
                print(folder_size)


def reservoir_sampling(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = randint(0, i)
            if j < k:
                reservoir[j] = item
    reservoir = set(reservoir)
    sample = []
    for elem in stream:
        if elem in reservoir:
            sample.append(elem)
    return sample


def build_dataset(n, distro):
    points = []
    for _ in range(n):
        x, y = random(), random()
        points.append([x, y, distro(x, y)])
    return points


def random_frame_point():
    return choice(
        [
            [random(), 0],
            [0, random()],
            [random(), 1],
            [1, random()],
        ]
    )


def generate_halfplane(n):
    p0, p1 = None, None
    while True:
        p0, p1 = random_frame_point(), random_frame_point()
        if p0[0] != p1[0] and p0[1] != p1[1]:
            break
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    hyp = (dx * dx + dy * dy) ** 0.5
    wx = dy / hyp
    wy = -dx / hyp
    b = -(wx * p0[0] + wy * p0[1])

    def distro(x, y):
        return black_pixel if wx * x + wy * y + b >= 0 else white_pixel

    return build_dataset(n, distro)


def generate_halfplane_with_error(n, threshold):
    p0, p1 = None, None
    while True:
        p0, p1 = random_frame_point(), random_frame_point()
        if p0[0] != p1[0] and p0[1] != p1[1]:
            break
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    hyp = (dx * dx + dy * dy) ** 0.5
    wx = dy / hyp
    wy = -dx / hyp
    b = -(wx * p0[0] + wy * p0[1])

    def distro(x, y):
        black = wx * x + wy * y + b >= 0
        if abs(wx * x + wy * y + b) <= threshold:
            black = not black
        return black_pixel if black else white_pixel

    return build_dataset(n, distro)


def change_labels(points, k):
    n = len(points)
    for _ in range(k):
        index = randint(0, n - 1)
        points[index][2] = (
            black_pixel if points[index][2] != black_pixel else white_pixel
        )


def random_id(n):
    return "".join([str(choice(list(range(10)))) for _ in range(n)])


def save_dataset(points):
    n = len(points)
    filename = f"D{n}_{1}_{random_id(9)}"
    with open(f"{datasets_path}/{filename}", "w") as file:
        file.write(f"{n}\n")
        for x, y, pixel in points:
            file.write(f"{x} {y} {pixel}\n")


class Approximator:
    def __init__(self, epsilon=0.01, delta=0.01):
        self.epsilon = epsilon
        self.delta = delta
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n = len(X)
        with open("input.txt", "w") as file:
            file.write(str(n) + "\n")
            for i in range(n):
                file.write(
                    f"{X[i][0]} {X[i][1]} {(black_pixel if y[i] == -1 else white_pixel)}\n"
                )
        command = ["./approximate line.exe", self.epsilon, self.delta]
        command = [str(val) for val in command]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = result.stdout
        error_msg = result.stderr
        # print(f"Output:\n{output}")
        # print(f"Error:\n{error_msg}")
        output = output.split()[:3]
        output = [float(val) for val in output]
        self.coef_ = np.array([output[0], output[1]])
        self.intercept_ = output[2]
        return
        y_pred1 = self.predict(X)

        self.coef_ *= -1
        self.intercept_ *= -1
        y_pred2 = self.predict(X)

        def comp(y_pred):
            nonlocal y
            for i in range(len(y_pred)):
                yield int(y_pred[i] == y[i])

        if sum(comp(y_pred1)) > sum(comp(y_pred2)):
            self.coef_ *= -1
            self.intercept_ *= -1

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("You  fit the model before making predictions.")
        scores = np.dot(X, self.coef_) + self.intercept_
        y_pred = np.sign(scores).astype(int)
        return y_pred

    def decision_function(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError(
                "You must fit the model before using the decision function."
            )
        scores = np.dot(X, self.coef_) + self.intercept_
        return scores
