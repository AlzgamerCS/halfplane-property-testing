#include <bits\stdc++.h>
#include "utility.cpp"

using namespace std;

int main(int argc, char *argv[]) // "./approximate.exe" "file name" "epsilon (sample)" "delta"
{
    double epsilon;
    double delta;
    vector<tuple<double, double, char>> points;
    if (argc == 4)
    {
        string filename = argv[1];
        epsilon = stod(argv[2]);
        delta = stod(argv[3]);
        auto tokens = split(filename, '_');
        int n = stoi(tokens[0]);
        double actual_distance = stod(tokens[1]);
        points.reserve(n);

        if (filename[0] == 'D')
        {
            ifstream file(datasets_path + "/" + filename);
            for (int i = 0; i < n; i++)
            {
                double x, y;
                char label;
                file >> x >> y >> label;
                points.push_back(make_tuple(x, y, label));
            }
        }
        else
        {
            ifstream file(images_path + "/" + filename);
            file >> n;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    char c;
                    file >> c;
                    points.push_back(make_tuple(i, j, c));
                }
            }
        }
    }
    else
    {
        epsilon = stod(argv[1]);
        delta = stod(argv[2]);
        ifstream file("input.txt");
        int n;
        file >> n;
        points.reserve(n);
        for (int i = 0; i < n; i++)
        {
            double x, y;
            char label;
            file >> x >> y >> label;
            points.push_back(make_tuple(x, y, label));
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto weight_bias = approximate_line(points, epsilon, delta);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    cout << weight_bias[0] << " " << weight_bias[1] << " " << weight_bias[2] << endl;
    cout << weight_bias[3] << endl;
    cout << duration.count() << std::endl;
    return 0;
}
