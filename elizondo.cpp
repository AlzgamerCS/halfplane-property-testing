#include <bits\stdc++.h>
#include "utility.cpp"

using namespace std;

int main(int argc, char *argv[]) // "./elizondo.exe" "number of tests" "file name"
{
    int tests = stoi(argv[1]);
    string file_name = argv[2];

    ifstream file(images_path + "/" + file_name);

    auto tokens = split(file_name, '_');
    double actual_distance = stod(tokens[1]);
    int n;
    file >> n;

    char *matrix[n];
    for (int i = 0; i < n; i++)
        matrix[i] = new char[n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            char c;
            while ((c = file.get()) == '\n')
                continue;
            matrix[i][j] = c;
        }
    }

    bool is_actually_halfplane = actual_distance == 0;

    double correct_guesses = 0;
    double total_time = 0;

    for (int i = 0; i < tests; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        bool is_halfplane = elizondo(matrix, n);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        total_time += duration.count();
        if (is_halfplane == is_actually_halfplane)
        {
            correct_guesses++;
        }
    }
    cout << correct_guesses / tests << endl;
    cout << total_time << endl;

    for (int i = 0; i < n; i++)
        delete[] matrix[i];

    return 0;
}
