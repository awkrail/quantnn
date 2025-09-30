#include <iostream>
#include <vector>

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


int MnistConv::forward(const std::vector<float> & data)
{
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
