#include <bits\stdc++.h>
#include "utility.cpp"
#include "svm.h"

using namespace std;

struct DataPoint
{
    vector<double> features;
    double label;
};

struct SVMModel
{
    svm_model *model;
};

svm_problem createSVMProblem(const vector<DataPoint> &data)
{
    svm_problem prob;
    prob.l = data.size();
    prob.y = new double[prob.l];
    prob.x = new svm_node *[prob.l];

    for (int i = 0; i < prob.l; ++i)
    {
        prob.y[i] = data[i].label;
        prob.x[i] = new svm_node[data[i].features.size() + 1];
        for (size_t j = 0; j < data[i].features.size(); ++j)
        {
            prob.x[i][j].index = j + 1;
            prob.x[i][j].value = data[i].features[j];
        }
        prob.x[i][data[i].features.size()].index = -1; // End of features
    }

    return prob;
}

void freeSVMProblem(svm_problem &prob)
{
    for (int i = 0; i < prob.l; ++i)
    {
        delete[] prob.x[i];
    }
    delete[] prob.x;
    delete[] prob.y;
}

void trainSVM(const vector<DataPoint> &data, SVMModel &model)
{
    svm_problem prob = createSVMProblem(data);

    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.C = 1e10; // Large C value for hard margin
    param.eps = 1e-3;

    // Default parameters
    param.degree = 3;
    param.gamma = 0;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;

    const char *error_msg = svm_check_parameter(&prob, &param);
    if (error_msg)
    {
        cerr << "Error: " << error_msg << endl;
        return;
    }

    model.model = svm_train(&prob, &param);
    freeSVMProblem(prob);
}

double predict(const SVMModel &model, const vector<double> &features)
{
    svm_node *x = new svm_node[features.size() + 1];
    for (size_t i = 0; i < features.size(); ++i)
    {
        x[i].index = i + 1;
        x[i].value = features[i];
    }
    x[features.size()].index = -1; // End of features

    double prediction = svm_predict(model.model, x);
    delete[] x;
    return prediction;
}

int main()
{
    // Example data
    vector<DataPoint> data = {
        {{2.0, 3.0}, 1},
        {{1.0, 1.0}, -1}};

    SVMModel model;
    trainSVM(data, model);

    vector<double> testPoint = {1.5, 2.0};
    double prediction = predict(model, testPoint);

    cout << "Prediction: " << prediction << endl;

    svm_free_and_destroy_model(&model.model);
    return 0;
}

int main(int argc, char *argv[]) // "./approximate.exe" "file name" "C - parameter"
{
    string file_name = argv[1];
    double c_param = stod(argv[2]);

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

    print_image(matrix, n);

    // Number of data points
    int numDataPoints = n * n;

    // Number of features (excluding bias)
    int numFeatures = 2;

    // Prepare the problem
    svm_problem problem;
    problem.l = numDataPoints;
    problem.y = new double[numDataPoints];
    problem.x = new svm_node *[numDataPoints];

    // Sample data points and labels
    vector<vector<double>> data;

    vector<double> labels;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            data.push_back({double(i) / (n - 1), double(j) / (n - 1)});
            labels.push_back(matrix[i][j] == black_pixel ? 1 : -1);
        }
    }

    // Populate problem data
    for (int i = 0; i < numDataPoints; i++)
    {
        problem.y[i] = labels[i];
        problem.x[i] = new svm_node[numFeatures + 1];
        for (int j = 0; j < numFeatures; j++)
        {
            problem.x[i][j].index = j + 1;
            problem.x[i][j].value = data[i][j];
        }
        problem.x[i][numFeatures].index = -1; // End of feature vector
    }

    // Set parameters
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.C = c_param;
    param.eps = 0.001;
    param.probability = 0;
    param.cache_size = 100;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;
    param.shrinking = 1;

    // Check parameter validity
    const char *error_msg = svm_check_parameter(&problem, &param);
    if (error_msg)
    {
        cerr << "Error: " << error_msg << endl;
        return 1;
    }

    // Train the model
    svm_model *model = svm_train(&problem, &param);

    if (model->param.kernel_type == LINEAR)
    {
        vector<double> weights(numFeatures, 0.0);
        for (int i = 0; i < model->l; i++)
        {
            double coef = model->sv_coef[0][i];
            for (int j = 0; j < numFeatures; j++)
                weights[j] += coef * model->SV[i][j].value;
        }

        for (int i = 0; i < numFeatures; i++)
            cout << weights[i] << endl;
        cout << -model->rho[0] << endl;
    }

    // Clean up
    for (int i = 0; i < numDataPoints; i++)
    {
        delete[] problem.x[i];
    }
    delete[] problem.y;
    delete[] problem.x;
    svm_free_and_destroy_model(&model);
    svm_destroy_param(&param);

    for (int i = 0; i < n; i++)
        delete[] matrix[i];
    return 0;
}
