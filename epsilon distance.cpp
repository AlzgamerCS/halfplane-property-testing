#include <bits/stdc++.h>
#include "utility.cpp"
using namespace std;

int main()
{
    int n;
    cin >> n;
    points data_points(n);
    for (int i = 0; i < n; i++)
    {
        cin >> data_points[i].x >> data_points[i].y >> data_points[i].label;
        data_points[i].label = data_points[i].label == black_pixel;
        // cout << data_points[i].x << " " << data_points[i].y << " " << int(data_points[i].label) << endl;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto result = double(epsilon_distance(data_points)) / double(n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    cout << result << endl;
    cout << duration.count() << endl;
    return 0;
}