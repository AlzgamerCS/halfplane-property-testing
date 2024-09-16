#include <bits\stdc++.h>
#include "utility.cpp"

using namespace std;

int main(int argc, char *argv[]) // "./tester.exe" "image size"
{
	int n = stoi(argv[1]);
	char *matrix[n];
	for (int i = 0; i < n; i++)
	{
		matrix[i] = new char[n];
	}

	int rand_choice = uniform_int_distribution<int>(0, 2)(gen);
	int abs_dist = 0;
	if (rand_choice == 0)
	{
		build_epsilon_close(matrix, n, uniform_real_distribution<double>(0.0, 0.5)(gen));
		abs_dist = epsilon_distance(matrix, n);
	}
	else if (rand_choice == 1)
	{
		build_gradient(matrix, n, [&](double x)
					   { return sigmoid(x / sqrt(n)); });
		abs_dist = epsilon_distance(matrix, n);
	}
	else
		build_halfplane(matrix, n);
	save_image(matrix, n, abs_dist);

	for (int i = 0; i < n; i++)
		delete[] matrix[i];
	return 0;
}
