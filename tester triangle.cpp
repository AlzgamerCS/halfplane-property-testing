#include <bits\stdc++.h>
#include "utility.cpp"

using namespace std;

int main(int argc, char *argv[]) // "./tester triangle.exe" "file name" "epsilon (sample)" "number of tests"
{
    double epsilon;
    double actual_distance;
    int tests = stoi(argv[3]);
    vector<tuple<double, double, char>> points;
    if (argc == 4)
    {
        string filename = argv[1];
        epsilon = stod(argv[2]);
        auto tokens = split(filename, '_');
        int n = stoi(tokens[0]);
        actual_distance = stod(tokens[1]);

        if (filename[0] == 'D')
        {
            points.reserve(n);
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
            points.reserve(n * n);
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
        ifstream file("input.txt");
        int n;
int main(int argc, char *argv[]) // "./tester triangle.exe" "epsilon (sample)" "number of tests" "file name"
{
    double epsilon = stod(argv[1]);
    int tests = stoi(argv[2]);
    string filename;
    int n;
    double actual_distance;
    ifstream file;
    vector<tuple<double, double, char>> points;
    if (argc == 4)
    {
        filename = argv[3];
        string filename = argv[1];
        auto tokens = split(filename, '_');
        actual_distance = stod(tokens[1]);
    }
    if (argc == 3 or filename[0] == 'D')
    {
        if (argc == 3)
            file = ifstream("input.txt");
        else
            file = ifstream(datasets_path + "/" + filename);
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

    bool is_actually_halfplane = actual_distance == 0;

    double correct_guesses = 0;
    double total_time = 0;

    for (int i = 0; i < tests; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        bool is_halfplane = triangle_test(points, 1 / epsilon);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        total_time += duration.count();
        if (is_halfplane == is_actually_halfplane)
            correct_guesses++;
    }
    cout << correct_guesses / tests << endl;
    cout << total_time << endl;
    return 0;
}
