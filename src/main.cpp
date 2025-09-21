#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "mnist_fc.h"
#include "data_7.h"


class MnistFC
{
public:
    MnistFC(const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias, 
            const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias);

    int forward(const std::vector<float> & data);
    void fc1(std::vector<float> & hidden, const std::vector<float> & data);
    void relu(std::vector<float> & hidden);
    void fc2(std::vector<float> & output, const std::vector<float> & hidden);

public:
    const std::vector<float> & fc1_weight;
    const std::vector<float> & fc1_bias;
    const std::vector<float> & fc2_weight;
    const std::vector<float> & fc2_bias;
    
    int input_dim = 784;
    int hidden_dim = 128;
    int output_dim = 10;
};


MnistFC::MnistFC(const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias,
                 const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias) : 
    fc1_weight{fc1_weight}, fc1_bias{fc1_bias}, fc2_weight{fc2_weight}, fc2_bias{fc2_bias} {}


int MnistFC::forward(const std::vector<float> & data)
{
    // fc1 + relu
    std::vector<float> hidden(hidden_dim);
    fc1(hidden, data);
    relu(hidden);

    // fc2 + relu
    std::vector<float> output(output_dim);
    fc2(output, hidden);

    int max_index = 0;
    float max_value = output[0];
    for (int i = 0; i < output_dim; i++)
    {
        if (output[i] > max_value)
        {
            max_index = i;
            max_value = output[i];
        }
    }

    return max_index;
}


void MnistFC::fc1(std::vector<float> & hidden, const std::vector<float> & data)
{
    for (int i = 0; i < hidden_dim; i++)
    {
        float value = 0;
        for (int j = 0; j < input_dim; j++)
        {
            value += fc1_weight[i * input_dim + j] * data[j];
        }
        hidden[i] = value + fc1_bias[i];
    }
}


void MnistFC::relu(std::vector<float> & hidden)
{
    for (int i = 0; i < hidden_dim; i++)
    {
        hidden[i] = std::max(0.0f, hidden[i]);
    }
}


void MnistFC::fc2(std::vector<float> & output, const std::vector<float> & hidden)
{
    for (int i = 0; i < output_dim; i++)
    {
        float value = 0;
        for (int j = 0; j < hidden_dim; j++)
        {
            value += fc2_weight[i * hidden_dim + j] * hidden[j];
        }
        output[i] = value + fc2_bias[i];
    }
}


int main(int argc, char * argv[])
{
    MnistFC model(fc1_weight, fc1_bias, fc2_weight, fc2_bias);
    int out = model.forward(data);
    std::cout << "Prediction: " << out << std::endl;
    return 0;
}
