#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "mnist_conv.h"

#include "quantized_conv1.h"
#include "quantized_fc1.h"
#include "quantized_fc2.h"

#include "data_7.h"

template <typename T>
struct QuantizedChannelBuffer
{
    std::vector<T> q;
    std::vector<float> s;
};

template <typename T>
struct QuantizedBuffer
{
    std::vector<T> q;
    float s;
};

class MnistConv
{
public:
    MnistConv(const QuantizedChannelBuffer<int8_t> & qconv1, const std::vector<float> & conv1_bias,
              const QuantizedBuffer<int8_t> & qfc1, const std::vector<float> & fc1_bias,
              const QuantizedBuffer<int8_t> & qfc2, const std::vector<float> & fc2_bias);

    QuantizedBuffer<int8_t> quantize(const std::vector<float> & data);
    QuantizedBuffer<int8_t> conv1(QuantizedBuffer<int8_t> & data);
    int forward(const std::vector<float> & data);

public:
    const QuantizedChannelBuffer<int8_t> qconv1;
    const QuantizedBuffer<int8_t> qfc1;
    const QuantizedBuffer<int8_t> qfc2;

    const std::vector<float> conv1_bias;
    const std::vector<float> fc1_bias;
    const std::vector<float> fc2_bias;

    const int image_size = 28;
    const int padded_image_size = 30;
    const int input_channel_num = 1;
    const int output_channel_num = 5;
    const int kernel_size = 3;
    const int stride = 1;
    const int pad_size = 1;
    const int fc1_input_dim = output_channel_num * image_size * image_size;
    const int fc1_hidden_dim = 128;
    const int fc2_hidden_dim = 10;
};


MnistConv::MnistConv(const QuantizedChannelBuffer<int8_t> & qconv1, const std::vector<float> & conv1_bias,
                     const QuantizedBuffer<int8_t> & qfc1, const std::vector<float> & fc1_bias,
                     const QuantizedBuffer<int8_t> & qfc2, const std::vector<float> & fc2_bias) 
    : qconv1{qconv1}, conv1_bias{conv1_bias}, qfc1{qfc1}, fc1_bias{fc1_bias}, qfc2{qfc2}, fc2_bias{fc1_bias} {}


QuantizedBuffer<int8_t> MnistConv::quantize(const std::vector<float> & data)
{
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    float s = std::max(std::abs(max_val), std::abs(min_val)) / 127.0f;
    std::vector<int8_t> quantized (data.size());
    for (int i = 0; i < data.size(); i++)
    {
        float qval = std::clamp(std::round(data[i] / s), -127.0f, 127.0f);
        quantized[i] = static_cast<int8_t>(qval);
    }
    return QuantizedBuffer<int8_t> { quantized, s };
}


QuantizedBuffer<int8_t> MnistConv::conv1(QuantizedBuffer<int8_t> & data)
{
    const int output_size = image_size - kernel_size + 2 * pad_size + 1;
    std::vector<float> output (output_channel_num * output_size * output_size);
    const int oW_size = padded_image_size - kernel_size + 1;
    const int oH_size = padded_image_size - kernel_size + 1;
    for (int o = 0; o < output_channel_num; o++)
    {
        for (int i = 0; i < oH_size; i++)
        {
            for (int j = 0; j < oW_size; j++)
            {
                int32_t qval = 0;
                for (int k = 0; k < kernel_size; k++)
                {
                    for (int l = 0; l < kernel_size; l++)
                    {
                        int target_index = (i + k) * padded_image_size + (j + l);
                        int weight_index = o * kernel_size * kernel_size + kernel_size * k + l;
                        qval += data.q[target_index] * qconv1.q[weight_index];
                    }
                }
                float rval = qval * qconv1.s[o] * data.s;
                int output_index = o * oH_size * oW_size + i * oW_size + j;
                output[output_index] = rval + conv1_bias[o];
            }
        }
    }
    
    // re-quantize
    float min_val = *std::min_element(output.begin(), output.end());
    float max_val = *std::max_element(output.begin(), output.end());
    float scale = std::max(std::abs(min_val), std::abs(max_val)) / 127.0f;
    std::vector<int8_t> quantized_output (output.size());
    for (int i = 0; i < output.size(); i++)
    {
        float qval = std::clamp(std::round(output[i] / scale), -127.0f, 127.0f);
        quantized_output[i] = static_cast<int8_t>(qval);
    }
    return QuantizedBuffer<int8_t> { quantized_output, scale };
}

int MnistConv::forward(const std::vector<float> & data)
{
    QuantizedBuffer<int8_t> qdata = quantize(data);
    for (int i = 0; i < qdata.q.size(); i++)
    {
        std::cout << (int)qdata.q[i] << std::endl;
    }
    qdata = conv1(qdata);
    return 0;
}

int main(int argc, char * argv[])
{
    const QuantizedChannelBuffer<int8_t> qconv1 { qconv1_weight, qconv1_scale };
    const QuantizedBuffer<int8_t> qfc1 { qfc1_weight, qfc1_scale };
    const QuantizedBuffer<int8_t> qfc2 { qfc2_weight, qfc1_scale };
    MnistConv model(qconv1, conv1_bias, qfc1, fc1_bias, qfc2, fc2_bias);
    int out = model.forward(data);
    std::cout << "Prediction: " << out << std::endl;
    return 0;
}
