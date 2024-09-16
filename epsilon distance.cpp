#include <bits/stdc++.h>
#include "utility.cpp"
using namespace std;

// struct pnt
// {
//     double x, y;
//     int c;
//     pnt(double x = 0, double y = 0, int c = 0) : x(x), y(y), c(c) {}
//     double l() const { return x * x + y * y; }
//     pnt operator+(const pnt a) { return pnt(x + a.x, y + a.y, c); }
//     pnt operator-(const pnt a) { return pnt(x - a.x, y - a.y, c); }
//     double operator*(const pnt a) const { return x * a.y - y * a.x; }
//     double operator%(const pnt a) const { return x * a.x + y * a.y; }
//     bool operator<(const pnt a)
//     {
//         if (x == a.x)
//             return y < a.y;
//         return x < a.x;
//     }
//     bool operator==(const pnt a) { return x == a.x && y == a.y; }
//     int pos() const
//     {
//         if (y > 0 || (y == 0 && x > 0))
//             return 0;
//         return 1;
//     }
// };

// bool cmp(const pnt &a, const pnt &b)
// {
//     if (a.pos() == b.pos())
//         return a * b > 0 || (a * b == 0 && a.l() < b.l());
//     return a.pos() < b.pos();
// }

// bool check(const pnt &a, const pnt &b)
// {
//     double v = a * b;
//     return v > 0 || (v == 0 && (a % b > 0));
// }

int main()
{
    // ios_base::sync_with_stdio(0), cin.tie(0);
    int n;
    cin >> n;
    vector<tuple<double, double, char>> data_points;
    data_points.reserve(n);
    for (int i = 0; i < n; i++)
    {
        double x, y;
        char label;
        cin >> x >> y >> label;
        data_points.push_back(make_tuple(x, y, label));
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto result = double(epsilon_distance(data_points)) / double(n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    cout << result << endl;
    // cout << duration.count() << endl;
    return 0;
}