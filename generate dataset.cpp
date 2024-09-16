#include <bits/stdc++.h>
#include "utility.cpp"

using namespace std;

int main(int argc, char *argv[]) // "./generate dataset.exe" "number of points" "to_change"
{
    int n = stoi(argv[1]);
    double to_change = stod(argv[2]);
    auto points = generate_dataset(n, to_change);
    auto dist = epsilon_distance(points);
    save_dataset(points, dist);
}