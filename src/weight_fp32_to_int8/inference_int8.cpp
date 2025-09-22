#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#include "quantized_fc1.h"
#include "quantized_fc2.h"
#include "data_7.h"

struct QuantizedBuffer
{
    std::vector<int8_t> q;
    float s;
};

class MnistFC
{
public:
    MnistFC(const QuantizedBuffer & qfc1, const std::vector<float> & fc1_bias,
            const QuantizedBuffer & qfc2, const std::vector<float> & fc2_bias);

    QuantizedBuffer quantize(const std::vector<float> & data);

    // int8 weight -> float32
    int forward_fp32(const std::vector<float> & data);
    void fc1(std::vector<float> & hidden, const std::vector<float> & data);
    void relu(std::vector<float> & hidden);
    void fc2(std::vector<float> & output, const std::vector<float> & hidden);

public:
    const QuantizedBuffer & qfc1;
    const QuantizedBuffer & qfc2;
    const std::vector<float> & fc1_bias;
    const std::vector<float> & fc2_bias;

    int input_dim = 784;
    int hidden_dim = 128;
    int output_dim = 10;

};

MnistFC::MnistFC(const QuantizedBuffer & qfc1, const std::vector<float> & fc1_bias,
                 const QuantizedBuffer & qfc2, const std::vector<float> & fc2_bias) 
    : qfc1{qfc1}, fc1_bias{fc1_bias}, qfc2{qfc2}, fc2_bias{fc2_bias} {}


int MnistFC::forward_fp32(const std::vector<float> & data)
{
    // convert int8 weight to float32 and calculate for checking
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
    std::vector<float> fc1_weight(qfc1.q.size());
    // convert int8 weight into float32
    for (int i = 0; i < fc1_weight.size(); i++)
    {
        fc1_weight[i] = static_cast<float>(qfc1.q[i]) * qfc1.s;
    }

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
    std::vector<float> fc2_weight(qfc2.q.size());
    for (int i = 0; i < fc2_weight.size(); i++)
    {
        fc2_weight[i] = static_cast<float>(qfc2.q[i]) * qfc2.s;
    }

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

QuantizedBuffer MnistFC::quantize(const std::vector<float> & data)
{
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    float s = (max_val - min_val) / 255.0f;
    
    std::vector<int8_t> quantized (data.size());
    for (int i = 0; i < data.size(); i++)
    {
        float clamped_qw = std::clamp(std::round(data[i] / s), -128.0f, 127.0f);
        quantized[i] = static_cast<int8_t>(clamped_qw);
    }

    return QuantizedBuffer { quantized, s };
}


int main(int argc, char * argv [])
{
    QuantizedBuffer qfc1 { fc1_weight, fc1_scale };
    QuantizedBuffer qfc2 { fc2_weight, fc2_scale };
    MnistFC model(qfc1, fc1_bias, qfc2, fc2_bias);
    int out = model.forward_fp32(data);
    std::cout << "Prediction: " << out << std::endl;
    return 0;
}
