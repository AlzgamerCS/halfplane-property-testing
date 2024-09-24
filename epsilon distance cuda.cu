#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <random>
#include <chrono>
#include <functional>
#include <cmath>
#include <math.h>
#include <numeric>
#include <climits>
#include <numbers>
#include <limits>
#include <cuda_runtime.h>

using namespace std;

char black_pixel = '#'; // 1 or true
char white_pixel = '.'; // 0 or false

template <typename T = double>
struct Point2D
{
    T x, y;
    char label;

    Point2D(T x = 0, T y = 0, char label = 0) : x(x), y(y), label(label) {}

    // Point2D operator+(const Point2D &other) const { return Point2D(x + other.x, y + other.y, label); }
    // Point2D operator-(const Point2D &other) const { return Point2D(x - other.x, y - other.y, label); }
    // void operator+=(const Point2D &other)
    // {
    //     x += other.x;
    //     y += other.y;
    // }
    // void operator-=(const Point2D &other)
    // {
    //     x -= other.x;
    //     y -= other.y;
    // }
    // T operator*(const Point2D &other) const { return x * other.y - y * other.x; }
    // T operator%(const Point2D &other) const { return x * other.x + y * other.y; }
    // int pos() const { return !(y > 0 or (y == 0 and x > 0)); }
    // bool operator<(const Point2D &other) const { return (x == other.x) ? y < other.y : x < other.x; }
    // bool operator==(const Point2D &other) const { return x == other.x and y == other.y; }
    // double l() const { return x * x + y * y; }

    friend ostream &operator<<(ostream &out, const Point2D<T> &p) { return out << "(" << p.x << ", " << p.y << ", " << int(p.label) << ")"; }
    friend istream &operator>>(istream &in, Point2D<T> &p) { return in >> p.x >> p.y >> p.label; }
};

using pt = Point2D<double>;
using points = vector<pt>;

__global__ void epsilon_distance_kernel(const pt *data_points, int n, int *results)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n && i != j)
    {
        int black_left = 0, black_right = 0, white_left = 0, white_right = 0;
        const auto &p1 = data_points[i], &p2 = data_points[j];
        double w0 = p1.y - p2.y, w1 = p2.x - p1.x;
        double b = p1.x * p2.y - p2.x * p1.y;
        for (int k = 0; k < n; k++)
        {
            const auto &p = data_points[k];
            double val = w0 * p.x + w1 * p.y + b;
            if (p.label == 1)
            {
                if (val > 0)
                    black_left++;
                else
                    black_right++;
            }
            else
            {
                if (val > 0)
                    white_left++;
                else
                    white_right++;
            }
        }
        results[i * n + j] = min(black_left + white_right, black_right + white_left);
    }
}

int epsilon_distance(const pt *data_points, const int n) // Calculate epsilon distane of a dataset to being a linearly separable
{
    pt *d_data_points;
    cudaMalloc((void **)&d_data_points, n * sizeof(pt));
    cudaMemcpy(d_data_points, data_points, n * sizeof(pt), cudaMemcpyHostToDevice);

    auto results = new int[n * n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            results[i * n + j] = n;

    int *d_results;
    cudaMalloc((void **)&d_results, n * n * sizeof(int));
    cudaMemcpy(d_results, results, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(33, 33);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    epsilon_distance_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data_points, n, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(results, d_results, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    int dist = n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dist = min(dist, results[i * n + j]);

    delete[] results;
    cudaFree(d_data_points);
    cudaFree(d_results);

    return dist;
}

int main()
{
    int n;
    cin >> n;
    auto data_points = new pt[n];
    for (int i = 0; i < n; i++)
    {
        cin >> data_points[i];
        data_points[i].label = data_points[i].label == black_pixel;
        // if (i < 10)
        //     cout << data_points[i] << endl;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto result = epsilon_distance(data_points, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    cout << result << endl;
    cout << duration.count() << endl;
    delete[] data_points;
    return 0;
}