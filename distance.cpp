#include <bits\stdc++.h>
#include "utility.cpp"

using namespace std;

int main(int argc, char *argv[]) // "./tester.exe" "file name"
{
    string file_name = argv[1];

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

    int abs_dist = epsilon_distance(matrix, n);

    cout << abs_dist;

    for (int i = 0; i < n; i++)
        delete[] matrix[i];
    return 0;
}
