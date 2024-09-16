#include <bits\stdc++.h>
#include "utility.cpp"

using namespace std;

int main(int argc, char *argv[])
{
    string filename = argv[1];
    ifstream file(images_path + "/" + filename);
    char c;
    while (file >> c)
    {
        if (c == EOF)
            break;
        cout << c;
    }
    cout << "Finish" << endl;
}