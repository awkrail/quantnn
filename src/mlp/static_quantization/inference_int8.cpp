#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdint>

#include "data_7.h"

#include "quantized_fc1.h"
#include "quantized_fc2.h"

// copied from the output of ./build/calibration
float input_scale = 0.0222164;
float fc1_output_scale = 0.171222;
float relu_output_scale = 0.0829648;

// fc1/fc2 -> int8_t, relu -> uint8_t
template<typename T>
struct QuantizedBuffer
{
    std::vector<T> q;
    float s;
};

class MnistFC
{
public:
    MnistFC(const QuantizedBuffer<int8_t> & qfc1, const std::vector<float> & fc1_bias,
            const QuantizedBuffer<int8_t> & qfc2, const std::vector<float> & fc2_bias);

    QuantizedBuffer<int8_t> quantize_int8(const std::vector<float> & data);
    QuantizedBuffer<int8_t> fc1(QuantizedBuffer<int8_t> & qinput);
    QuantizedBuffer<uint8_t> relu(QuantizedBuffer<int8_t> & hidden);
    int fc2(QuantizedBuffer<uint8_t> & hidden);

    int forward_int8(const std::vector<float> & data);

public:
    const QuantizedBuffer<int8_t> & qfc1;
    const QuantizedBuffer<int8_t> & qfc2;
    const std::vector<float> & fc1_bias;
    const std::vector<float> & fc2_bias;

    int input_dim = 784;
    int hidden_dim = 128;
    int output_dim = 10;
};

MnistFC::MnistFC(const QuantizedBuffer<int8_t> & qfc1, const std::vector<float> & fc1_bias,
                 const QuantizedBuffer<int8_t> & qfc2, const std::vector<float> & fc2_bias)
    : qfc1{qfc1}, fc1_bias{fc1_bias}, qfc2{qfc2}, fc2_bias{fc2_bias} {}

QuantizedBuffer<int8_t> MnistFC::quantize_int8(const std::vector<float> & data)
{
    std::vector<int8_t> quantized (data.size());
    for (int i = 0; i < data.size(); i++)
    {
        float qval = std::clamp(std::round(data[i] / input_scale), -127.0f, 127.0f);
        quantized[i] = static_cast<int8_t>(qval);
    }
    return QuantizedBuffer<int8_t> { quantized, input_scale };
}

QuantizedBuffer<int8_t> MnistFC::fc1(QuantizedBuffer<int8_t> & qinput)
{
    // quantize fc1_bias
    float scale = qinput.s * qfc1.s;
    std::vector<int32_t> bias_int32 (fc1_bias.size());
    for (int i = 0; i < fc1_bias.size(); i++)
    {
        bias_int32[i] = static_cast<int32_t>(std::round(fc1_bias[i] / scale));
    }

    // int8 calculation
    std::vector<int8_t> hidden (hidden_dim);
    for (int i = 0; i < hidden_dim; i++)
    {
        int32_t value = 0;
        for (int j = 0; j < input_dim; j++)
        {
            value += static_cast<int32_t>(qfc1.q[i * input_dim + j]) * static_cast<int32_t>(qinput.q[j]);
        }
        value = value + bias_int32[i];
        float qval = std::round((value * scale) / fc1_output_scale);
        hidden[i] = static_cast<int8_t>(std::clamp(qval, -127.0f, 127.0f));
    }

    return QuantizedBuffer<int8_t> { hidden, fc1_output_scale };
}

QuantizedBuffer<uint8_t> MnistFC::relu(QuantizedBuffer<int8_t> & hidden)
{
    std::vector<uint8_t> relu_hidden (hidden.q.size());
    for (int i = 0; i < hidden.q.size(); i++)
    {
        float val = static_cast<float>(hidden.q[i]) * hidden.s;
        float qval = std::clamp(std::round(val / relu_output_scale), 0.0f, 255.0f);
        relu_hidden[i] = static_cast<uint8_t>(qval);
    }

    return QuantizedBuffer<uint8_t> { relu_hidden, relu_output_scale };
}

int MnistFC::fc2(QuantizedBuffer<uint8_t> & hidden)
{
    // quantize fc2_bias
    float scale = hidden.s * qfc2.s;
    std::vector<int32_t> bias_int32 (fc2_bias.size());
    for (int i = 0; i < fc2_bias.size(); i++)
    {
        bias_int32[i] = static_cast<int32_t>(std::round(fc2_bias[i] / scale));
    }

    // int8 calculation
    std::vector<int32_t> output (output_dim);
    for (int i = 0; i < output_dim; i++)
    {
        int32_t value = 0;
        for (int j = 0; j < hidden_dim; j++)
        {
            value += static_cast<int32_t>(qfc2.q[i * hidden_dim + j]) * static_cast<int32_t>(hidden.q[j]);
        }
        output[i] = value + bias_int32[i];
    }

    // output prediction_idx
    auto it = std::max_element(output.begin(), output.end());
    int max_index = std::distance(output.begin(), it);
    return max_index;
}

int MnistFC::forward_int8(const std::vector<float> & data)
{
    QuantizedBuffer<int8_t> qinput = quantize_int8(data);
    QuantizedBuffer<int8_t> hidden = fc1(qinput);
    QuantizedBuffer<uint8_t> relu_hidden = relu(hidden);
    int prediction = fc2(relu_hidden);
    return prediction;
}

int main(int argc, char * argv [])
{
    QuantizedBuffer<int8_t> qfc1 { fc1_weight, fc1_scale };
    QuantizedBuffer<int8_t> qfc2 { fc2_weight, fc2_scale };
    MnistFC model(qfc1, fc1_bias, qfc2, fc2_bias);
    int out = model.forward_int8(data);
    std::cout << "Prediction: " << out << std::endl;
    return 0;
}
