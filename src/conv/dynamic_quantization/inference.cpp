#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string.h>

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
    int zp;
};

class MnistConv
{
public:
    MnistConv(const QuantizedChannelBuffer<int8_t> & qconv1, const std::vector<float> & conv1_bias,
              const QuantizedBuffer<int8_t> & qfc1, const std::vector<float> & fc1_bias,
              const QuantizedBuffer<int8_t> & qfc2, const std::vector<float> & fc2_bias);

    std::vector<float> padding(std::vector<float> & data);
    QuantizedBuffer<int8_t> quantize(const std::vector<float> & data);
    QuantizedBuffer<uint8_t> quantize_uint8(const std::vector<float> & data);
    QuantizedBuffer<int8_t> conv1(QuantizedBuffer<int8_t> & data);
    QuantizedBuffer<int8_t> fc1(QuantizedBuffer<int8_t> & data);
    QuantizedBuffer<uint8_t> relu(QuantizedBuffer<int8_t> & data);
    std::vector<float> fc2(QuantizedBuffer<uint8_t> & data);
    int forward(std::vector<float> & data);

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
    : qconv1{qconv1}, conv1_bias{conv1_bias}, qfc1{qfc1}, fc1_bias{fc1_bias}, qfc2{qfc2}, fc2_bias{fc2_bias} {}

std::vector<float> MnistConv::padding(std::vector<float> & data)
{
    std::vector<float> padded_data(padded_image_size * padded_image_size, 0.0f);
    for (int i = 0; i < image_size; i++)
    {
        float * dst = &padded_data[(i + pad_size) * padded_image_size + pad_size];
        float * src = &data[i * image_size];
        memcpy(dst, src, image_size * sizeof(float));
    }
    return padded_data;
}

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
    return QuantizedBuffer<int8_t> { quantized, s, 0 };
}

QuantizedBuffer<uint8_t> MnistConv::quantize_uint8(const std::vector<float> & data)
{
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    float s = (max_val - min_val) / 255.0f;
    int zp = static_cast<int>(std::round(-min_val / s));
    zp = std::clamp(zp, 0, 255);
    std::vector<uint8_t> quantized (data.size());
    for (int i = 0; i < data.size(); i++)
    {
        int qval = static_cast<int>(std::round(data[i] / s)) + zp;
        qval = std::clamp(qval, 0, 255);
        quantized[i] = static_cast<uint8_t>(qval);
    }
    return QuantizedBuffer<uint8_t> { quantized, s, zp };
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
                        qval += static_cast<int32_t>(data.q[target_index]) * static_cast<int32_t>(qconv1.q[weight_index]);
                    }
                }
                float rval = qconv1.s[o] * data.s * qval + conv1_bias[o];
                int output_index = o * oH_size * oW_size + i * oW_size + j;
                output[output_index] = rval;
            }
        }
    }
    return quantize(output);
}

QuantizedBuffer<int8_t> MnistConv::fc1(QuantizedBuffer<int8_t> & data)
{
    std::vector<float> output (fc1_hidden_dim);
    for (int i = 0; i < fc1_hidden_dim; i++)
    {
        int32_t qval = 0;
        for (int j = 0; j < fc1_input_dim; j++)
        {
            qval += static_cast<int32_t>(qfc1.q[i * fc1_input_dim + j]) * static_cast<int32_t>(data.q[j]);
        }
        float value = data.s * qfc1.s * qval + fc1_bias[i];
        output[i] = value;
    }
    return quantize(output);
}

std::vector<float> MnistConv::fc2(QuantizedBuffer<uint8_t> & data)
{
    std::vector<float> output (fc2_hidden_dim);
    for (int i = 0; i < fc2_hidden_dim; i++)
    {
        int32_t qval = 0;
        for (int j = 0; j < fc1_hidden_dim; j++)
        {
            qval += static_cast<int32_t>(qfc2.q[i * fc1_hidden_dim + j]) * static_cast<int32_t>(data.q[j]);
        }
        float value = data.s * qfc2.s * qval + fc2_bias[i];
        output[i] = value;
    }
    return output;
}

QuantizedBuffer<uint8_t> MnistConv::relu(QuantizedBuffer<int8_t> & data)
{
    std::vector<float> output (fc1_hidden_dim);
    for (int i = 0; i < fc1_hidden_dim; i++)
    {
        float value = static_cast<float>(data.q[i]) * data.s;
        output[i] = std::max(0.0f, value);
    }
    
    return quantize_uint8(output);
}

int MnistConv::forward(std::vector<float> & data)
{
    QuantizedBuffer<int8_t> qdata = quantize(padding(data));
    qdata = conv1(qdata);
    qdata = fc1(qdata);
    QuantizedBuffer<uint8_t> uint8_qdata = relu(qdata);
    //QuantizedBuffer<int8_t> uint8_qdata = relu(qdata);
    std::vector<float> output = fc2(uint8_qdata);

    int max_index = 0;
    float max_val = 1e-5;
    for (int i = 0; i < output.size(); i++)
    {
        if (max_val < output[i])
        {
            max_val = output[i];
            max_index = i;
        }
    }
    return max_index;
}

int main(int argc, char * argv[])
{
    const QuantizedChannelBuffer<int8_t> qconv1 { qconv1_weight, qconv1_scale };
    const QuantizedBuffer<int8_t> qfc1 { qfc1_weight, qfc1_scale };
    const QuantizedBuffer<int8_t> qfc2 { qfc2_weight, qfc2_scale };
    MnistConv model(qconv1, conv1_bias, qfc1, fc1_bias, qfc2, fc2_bias);
    int out = model.forward(data);
    std::cout << "Prediction: " << out << std::endl;
    return 0;
}
