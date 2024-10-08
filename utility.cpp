#pragma once

// #include <bits\stdc++.h>
// #include <format>

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

using namespace std;

#define M_PI 3.14159265358979323846
#define INF numeric_limits<double>::infinity()
using int_pair = pair<int, int>;
string images_path = "./Images";
string datasets_path = "./Datasets";

auto seed = random_device()();

mt19937_64 gen(seed);

uniform_real_distribution<> random_frac(0.0, 1.0);

char black_pixel = '#'; // 1 or true
char white_pixel = '.'; // 0 or false

template <typename T = double>
struct DataPoint
{
    vector<T> coords;
    char label;
};

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

vector<string> split(string &str, char delimiter)
{
    vector<string> tokens;
    string token;
    istringstream token_stream(str);
    while (getline(token_stream, token, delimiter))
        tokens.push_back(token);
    return tokens;
}

void print_image(char *matrix[], int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            cout << matrix[i][j];
        cout << "\n";
    }
}

string random_id(int n)
{
    string id(n, ' ');
    uniform_int_distribution<int> rand_digit(0, 9);
    for (int i = 0; i < n; i++)
        id[i] = char('0' + rand_digit(gen));
    return id;
}

void save_image(char *matrix[], int n, int abs_distance) // image -> txt file
{
    double dist = double(abs_distance) / double(n * n);
    string path = format("./Images/{}_{:.15f}_{}", n, dist, random_id(4));
    ofstream outfile(path);
    if (not outfile.is_open())
    {
        cerr << "Error: Unable to open the file." << endl;
        return;
    }
    outfile << n << "\n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            outfile << matrix[i][j];
        outfile << "\n";
    }
    outfile.close();
}

void save_dataset(vector<DataPoint<>> &points, int abs_distance) // dataset -> txt file
{
    int n = points.size();
    double dist = double(abs_distance) / double(n);
    string path = format("./Datasets/D{}_{:.15f}_{}", n, dist, random_id(4));
    ofstream outfile(path);
    if (not outfile.is_open())
    {
        cerr << "Error: Unable to open the file." << endl;
        return;
    }
    outfile << n << "\n";
    for (auto &p : points)
    {
        for (double x : p.coords)
            outfile << x << " ";
        outfile << p.label << "\n";
    }
    outfile.close();
}

int_pair get_normal(int_pair p1, int_pair p2)
{
    if (p1 == p2)
        return make_pair(1, 1);
    p2.first -= p1.first;
    p2.second -= p1.second;
    int d = gcd(abs(p2.first), abs(p2.second));
    return {-p2.second / d, p2.first / d};
}

#if 0

struct point_pair_hasher
{
    size_t operator()(pair<int_pair, int_pair> &point_pair) const
    {
        hash<int> h;
        return (h(point_pair.first.first) << 48) ^ (h(point_pair.first.second) << 32) ^ (h(point_pair.second.first) << 16) ^ h(point_pair.second.second);
    }
};

bool elizondo(char *matrix[], int n)
{                                    // checks if an image is halfplane
    int_pair a = {0, 0}, b = {0, 0}; // a - black, b - white
    int black_count = 0, white_count;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (matrix[i][j] == black_pixel)
            {
                black_count++;
                a.first += i;
                a.second += j;
            }
            else
            {
                b.first += i;
                b.second += j;
            }
        }
    }
    if (black_count == 0 or black_count == n * n)
        return true;
    white_count = n * n - black_count;

    a.first /= black_count; // gravity centers
    a.second /= black_count;
    b.first /= white_count;
    b.second /= white_count;

    // cout << format("({}, {}) ({}, {})\n", a.first, a.second, b.first, b.second);
    int_pair normal = get_normal(a, b);
    // cout << "Start" << endl;
    unordered_set<pair<int_pair, int_pair>, point_pair_hasher> visited_pairs;
    // int iter = 0;
    while (true)
    {
        // cout << iter << endl;
        // iter++;
        if (visited_pairs.count(make_pair(a, b)))
            return false;
        visited_pairs.insert(make_pair(a, b));

        int max_val = INT_MIN, min_val = INT_MAX;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int product = normal.first * i + normal.second * j;
                if (matrix[i][j] == black_pixel)
                {
                    if (product <= min_val)
                    {
                        min_val = product;
                        a.first = i;
                        a.second = j;
                    }
                }
                else
                {
                    if (product >= max_val)
                    {
                        max_val = product;
                        b.first = i;
                        b.second = j;
                    }
                }
            }
        }
        if (max_val <= min_val)
            break;
        normal = get_normal(a, b);
    }
    return true;
}

int epsilon_distance(char *matrix[], const int n) // Calculate epsilon distane of an image to being a halfplane
{
    int black_total = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (matrix[i][j] == black_pixel)
                black_total++;
        }
    }
    int white_total = n * n - black_total;
    vector<unsigned short int> white_counts(3 * n * n, 0);
    vector<unsigned short int> black_counts(3 * n * n, 0);
    int dist = n * n;
    for (int ny = 0; ny < n; ny++)
    {
        for (int nx = -(n - 1); nx < n; nx++)
        {
            if (gcd(ny, abs(nx)) != 1)
                continue;
            for (int y = 0; y < n; y++)
            {
                for (int x = 0; x < n; x++)
                {
                    int dot_product = ny * y + nx * x + n * n;
                    if (matrix[y][x] == black_pixel)
                        black_counts[dot_product]++;
                    else
                        white_counts[dot_product]++;
                }
            }

            int pre_black = 0, pre_white = 0;
            for (int prod = 0; prod < 3 * n * n; prod++)
            {
                pre_black += black_counts[prod];
                pre_white += white_counts[prod];
                dist = min({dist, pre_black + white_total - pre_white, pre_white + black_total - pre_black});
                black_counts[prod] = 0;
                white_counts[prod] = 0;
            }
        }
    }
    return dist;
}

int epsilon_distance(vector<DataPoint<>> &points) // Calculate epsilon distane of a dataset to being a linearly separable
{
    int n = points.size();
    int dist = n;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if(i == j)
                continue;
            // cout << i << " " << j << endl;
            int black_left = 0, black_right = 0, white_left = 0, white_right = 0;
            auto &p1 = points[i], &p2 = points[j];
            double w0 = get<1>(p1) - get<1>(p2), w1 = get<0>(p2) - get<0>(p1);
            double b = get<0>(p1) * get<1>(p2) - get<0>(p2) * get<1>(p1);
            for (int k = 0; k < n; k++)
            {
                auto &p = points[k];
                double val = w0 * get<0>(p) + w1 * get<1>(p) + b;
                if (get<2>(p) == black_pixel)
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
            dist = min({dist, black_left + white_right, black_right + white_left});
        }
    }
    return dist;
}

vector<double> approximate_line(vector<DataPoint<>> &points, double epsilon, double delta) // Approximate best separation line of an image, black = negative, white = positive
{
    if (delta == 0)
        delta = epsilon;
    uniform_int_distribution<int> rand_index(0, points.size() - 1);
    int sample_size = 1 / epsilon;
    if (sample_size > points.size())
        sample_size = points.size();
    int sample[sample_size];
    int black_total = 0;
    pair<double, double> black_center, white_center;
    black_center = white_center = make_pair(0, 0);
    for (int i = 0; i < sample_size; i++)
    {
        int index = i;
        if (sample_size < points.size())
            index = rand_index(gen);
        sample[i] = index;
        auto &point = points[index];
        if (get<2>(point) == black_pixel)
        {
            black_total++;
            black_center.first += get<0>(point);
            black_center.second += get<1>(point);
        }
        else
        {
            white_center.first += get<0>(point);
            white_center.second += get<1>(point);
        }
    }
    int white_total = sample_size - black_total;

    black_center.first /= black_total;
    black_center.second /= black_total;
    white_center.first /= white_total;
    white_center.second /= white_total;

    double delta_angle = 2 * M_PI * delta;
    int bucket_count = 1 / delta;
    int_pair buckets[bucket_count]; // first = black count, second = white count
    for (auto &count : buckets)
        count = make_pair(0, 0);
    int min_distance = sample_size;
    double min_diff = INF;
    pair<double, double> result_weight;
    double result_bias;

    for (double angle = 0; angle < 2 * M_PI; angle += delta_angle)
    {
        double min_prod = INF, max_prod = -INF;
        for (auto index : sample)
        {
            auto &point = points[index];
            double prod = get<0>(point) * cos(angle) + get<1>(point) * sin(angle);
            min_prod = min(min_prod, prod);
            max_prod = max(max_prod, prod);
        }
        double delta_thresh = (max_prod - min_prod) / (bucket_count - 1);
        for (auto index : sample)
        {
            auto &point = points[index];
            double prod = get<0>(point) * cos(angle) + get<1>(point) * sin(angle);
            int bucket = (prod - min_prod) / delta_thresh;
            if (get<2>(point) == black_pixel)
                buckets[bucket].first++;
            else
                buckets[bucket].second++;
        }

        int pre_black = 0, pre_white = 0;
        for (int bucket = 0; bucket < bucket_count; bucket++)
        {
            // cout << min_distance << endl;
            int distance = pre_white + black_total - pre_black;
            pair<double, double> weight = make_pair(cos(angle), sin(angle));
            double bias = -(bucket * delta_thresh + min_prod);
            double a = black_center.first * cos(angle) + black_center.second * sin(angle) + bias;
            double b = white_center.first * cos(angle) + white_center.second * sin(angle) + bias;
            double diff = abs(abs(a) - abs(b));
            if (distance < min_distance or (distance == min_distance and diff < min_diff))
            {
                min_distance = distance;
                min_diff = diff;
                result_weight = weight;
                result_bias = bias;
            }
            pre_black += buckets[bucket].first;
            pre_white += buckets[bucket].second;
            buckets[bucket].first = 0;
            buckets[bucket].second = 0;
        }
    }
    return {result_weight.first, result_weight.second, result_bias, double(min_distance) / double(sample_size)};
}

#endif

template <typename T = double>
struct Point2D
{
    T x, y;
    char label;

    Point2D(T x = 0, T y = 0, char label = 0) : x(x), y(y), label(label) {}

    Point2D operator+(const Point2D &other) const { return Point2D(x + other.x, y + other.y, label); }
    Point2D operator-(const Point2D &other) const { return Point2D(x - other.x, y - other.y, label); }
    void operator+=(const Point2D &other)
    {
        x += other.x;
        y += other.y;
    }
    void operator-=(const Point2D &other)
    {
        x -= other.x;
        y -= other.y;
    }
    T operator*(const Point2D &other) const { return x * other.y - y * other.x; }
    T operator%(const Point2D &other) const { return x * other.x + y * other.y; }
    int pos() const { return !(y > 0 or (y == 0 and x > 0)); }
    bool operator<(const Point2D &other) const { return (x == other.x) ? y < other.y : x < other.x; }
    bool operator==(const Point2D &other) const { return x == other.x and y == other.y; }
    double l() const { return x * x + y * y; }

    friend ostream &operator<<(ostream &out, const Point2D<T> &p) { return out << "(" << p.x << ", " << p.y << ", " << p.label << ")"; }
    friend istream &operator>>(istream &in, Point2D<T> &p) { return in >> p.x >> p.y >> p.label; }
};

using pt = Point2D<double>;
using points = vector<pt>;

void add(points &h, pt &p)
{
    if (h.size() <= 1)
    {
        h.push_back(p);
        return;
    }
    if (h.back() == p)
        return;
    while (h.size() > 1 && (h.end()[-1] - h.end()[-2]) * (p - h.end()[-1]) <= 0)
        h.pop_back();
    h.push_back(p);
    return;
}

points hull(points s)
{
    if (s.size() == 0)
        return s;
    sort(s.begin(), s.end());
    points lower, upper;
    for (auto p : s)
        add(lower, p);
    reverse(s.begin(), s.end());
    for (auto p : s)
        add(upper, p);
    lower.pop_back();
    for (auto p : upper)
        lower.push_back(p);
    lower.pop_back();
    return lower;
}

bool intersect(points &A, points &B)
{
    auto check = [](const pt &v, const pt &u)
    {
        if (v.pos() == u.pos())
            return v * u > 0;
        return v.pos() < u.pos();
    };
    if (A.empty() || B.empty())
        return false;
    points V;
    for (auto &p : B)
    {
        p.x *= -1;
        p.y *= -1;
    }
    rotate(B.begin(), min_element(B.begin(), B.end()), B.end());
    int a = 0, b = 0;
    while (a < int(A.size()) || b < int(B.size()))
    {
        pt v = (A[(a + 1) % A.size()] - A[(a) % A.size()]), u = (B[(b + 1) % B.size()] - B[(b) % B.size()]);
        if (a == int(A.size()))
        {
            V.push_back(u);
            b++;
            continue;
        }
        if (b == int(B.size()))
        {
            V.push_back(v);
            a++;
            continue;
        }
        if (check(v, u))
        {
            V.push_back(v);
            a++;
        }
        else
        {
            V.push_back(u);
            b++;
        }
    }
    pt mn = A[0] + B[0];
    for (auto p : V)
    {
        if (mn * p < 0)
            return false;
        mn = mn + p;
    }
    return true;
}

bool test_convex_hull(vector<tuple<double, double, char>> &data_points, double epsilon) // tests if the image is a halfplane
{
    int sample_count = ceil(1 / epsilon);

    points W, B;

    uniform_int_distribution<int> rand_index(0, data_points.size() - 1);
    for (int i = 0; i < sample_count; ++i)
    {
        int index = rand_index(gen);
        auto &p = data_points[index];
        if (get<2>(p) == white_pixel)
            W.push_back(pt(get<0>(p), get<1>(p)));
        else
            B.push_back(pt(get<0>(p), get<1>(p)));
    }
    W = hull(W);
    B = hull(B);
    return not intersect(W, B);
}

bool cmp(const pt &a, const pt &b)
{
    if (a.pos() == b.pos())
        return a * b > 0 || (a * b == 0 && a.l() < b.l());
    return a.pos() < b.pos();
}

bool check(const pt &a, const pt &b)
{
    double v = a * b;
    return v > 0 || (v == 0 && (a % b > 0));
}

int epsilon_distance(points &data_points)
{
    int n = data_points.size();
    int all = 0;
    for (auto &elem : data_points)
        all += elem.label;
    int ans = n;
    for (int l = 0; l < n; ++l)
    {
        points v = data_points;
        int m = n - 1;
        swap(v[m], v[l]);
        for (int j = 0; j < m; ++j)
            v[j] = v[j] - v[m];
        sort(v.begin(), v.end() - 1, cmp);
        int now = all;
        now += (v[m].label == 0) - (v[m].label == 1);
        for (int i = 0, j = -1; i < m; ++i)
        {
            j = max(j, i - 1);
            if (i > 0 && v[i] * v[i - 1] == 0 && v[i] % v[i - 1] > 0)
            {
                now -= (v[i].label == 0) - (v[i].label == 1);
                continue;
            }
            while (j + 2 - i <= m && check(v[i], v[(j + 1) % m]))
            {
                j++;
                now += (v[j % m].label == 0) - (v[j % m].label == 1);
            }
            ans = min(ans, now);
            ans = min(ans, n - now);
            now -= (v[i].label == 0) - (v[i].label == 1);
        }
        now = all;
        now += (v[m].label == 0) - (v[m].label == 1);
        for (int i = 0, j = -1; i < m; ++i)
        {
            j = max(j, i - 1);
            if (i > 0 && v[i] * v[i - 1] == 0 && v[i] % v[i - 1] > 0)
            {
                now -= (v[i].label == 0) - (v[i].label == 1);
                continue;
            }
            while (j + 2 - i <= m && v[i] * v[(j + 1) % m] >= 0)
            {
                j++;
                now += (v[j % m].label == 0) - (v[j % m].label == 1);
            }
            ans = min(ans, now);
            ans = min(ans, n - now);
            now -= (v[i].label == 0) - (v[i].label == 1);
        }
    }
    return ans;
}

#if 0
int epsilon_distance(points &data_points)
{
    int n = data_points.size();
    int black_total = 0;
    for (auto &elem : data_points)
        if (elem.label)
            black_total++;
    int white_total = n - black_total;
    int min_dist = n;
    for (int i = 0; i < n; i++)
    {
        auto pivot = data_points[i];
        auto get_angle = [&](pt p)
        {
            if (p == pivot)
                return INF;
            p -= pivot;
            return atan2(p.y, p.x);
        };
        auto sorted_points = data_points;
        sort(sorted_points.begin(), sorted_points.end(), [&](const pt &a, const pt &b)
             { 
                double angle1 = get_angle(a), angle2 = get_angle(b); 
                if(angle1 < 0)
                    angle1 += M_PI;
                if(angle2 < 0)
                    angle2 += M_PI;
                return angle1 < angle2; });
        sorted_points.pop_back();
        int white_left = 0, black_left = 0;
        pivot.label ? black_left++ : white_left++;
        for (auto &elem : sorted_points)
        {
            if (get_angle(elem) >= 0)
                elem.label ? black_left++ : white_left++;
        }
        min_dist = min({min_dist, black_left + white_total - white_left, white_left + black_total - black_left});
        double prev_angle = 0;
        for (auto &elem : sorted_points)
        {
            double angle = get_angle(elem);
            if (angle == prev_angle)
            {
            }
            else
            {
            }
            prev_angle = angle;
        }

    }
    return n;
}
#endif