#include <bits\stdc++.h>
#include "utility.cpp"

using namespace std;

int main(int argc, char *argv[]) // "./tester vs elizondo.exe" "file name" "epsilon" "number of tests"
{
    string file_name = argv[1];
    double epsilon = stod(argv[2]);
    int tests = stoi(argv[3]);

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

    cout << "Actual distance (ratio): " << actual_distance << endl;
    bool is_actually_halfplane = actual_distance == 0;
    cout << "Image is actually " << (is_actually_halfplane ? "" : "not ") << "halfplane\n"
         << endl;

    bool is_halfplane;
    int elizondo_corrent_guess = 0;
    int property_test_correct_guess = 0;
    double elizondo_time = 0;
    double property_test_time = 0;

    for (int i = 0; i < tests; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        is_halfplane = elizondo(matrix, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        elizondo_time += duration.count();
        if (is_actually_halfplane == is_halfplane)
        {
            elizondo_corrent_guess++;
        }

        start = std::chrono::high_resolution_clock::now();
        is_halfplane = property_test(matrix, n, epsilon);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        property_test_time += duration.count();
        if (is_actually_halfplane == is_halfplane)
        {
            property_test_correct_guess++;
        }
    }

    cout << "Elizondo: " << elizondo_time << " seconds" << endl;
    cout << "Correct Guesses: " << std::fixed << std::setprecision(9) << ((double)elizondo_corrent_guess) / tests * 100.0 << "%\n\n";
    cout << "Property tester: " << property_test_time << " seconds" << endl;
    cout << "Correct Guesses: " << std::fixed << std::setprecision(9) << ((double)property_test_correct_guess) / tests * 100.0 << "%\n\n";

    for (int i = 0; i < n; i++)
    {
        delete[] matrix[i];
    }
    return 0;
}
