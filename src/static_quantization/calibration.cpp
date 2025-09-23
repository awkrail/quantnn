#include <iostream>
#include <vector>

#include "calibration_data.h"
#include "mnist_fc.h"

struct Scale
{
    float input_scale;
    float fc1_scale;
    float relu_scale;
};

float calculate_scale_int8(const std::vector<std::vector<float>> & data)
{
    // calculate scale on input
    float min_val = 1e+5;
    float max_val = 1e-5;
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[i].size(); j++)
        {
            min_val = std::min(data[i][j], min_val);
            max_val = std::max(data[i][j], max_val);
        }
    }
    float scale = (max_val - min_val) / 254.0f;
    return scale;
}

class MnistFC
{
public:
    MnistFC(const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias,
            const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias); 

    Scale calibrate(const std::vector<std::vector<float>> & calibration_data);
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

Scale MnistFC::calibrate(const std::vector<std::vector<float>> & calibration_data)
{
    float input_scale = calculate_scale_int8(calibration_data);

    // fc1
    std::vector<std::vector<float>> hiddens (calibration_data.size());
    for (int i = 0; i < calibration_data.size(); i++)
    {
        std::vector<float> hidden(hidden_dim);
        fc1(hidden, calibration_data[i]);
        hiddens[i] = hidden;
    }
    float fc1_scale = calculate_scale_int8(hiddens);

    // relu

    return Scale { input_scale, fc1_scale, 0.0f };
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
    Scale model_scale = model.calibrate(calibration_data);

    float input_scale = model_scale.input_scale;
    float fc1_scale = model_scale.fc1_scale;
    float relu_scale = model_scale.relu_scale;

    std::cout << input_scale << " " << fc1_scale << " " << relu_scale << std::endl;

    return 0;
}
