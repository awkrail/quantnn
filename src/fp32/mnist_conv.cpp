#include <iostream>
#include <vector>

#include "mnist_conv.h"
#include "data_7.h"


class MnistConv
{
public:
    MnistConv(const std::vector<float> & conv1_weight, const std::vector<float> & conv1_bias,
              const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias,
              const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias);
    int forward(std::vector<float> & data);

public:
    const std::vector<float> & conv1_weight;
    const std::vector<float> & fc1_weight;
    const std::vector<float> & fc2_weight;

    const std::vector<float> & conv1_bias;
    const std::vector<float> & fc1_bias;
    const std::vector<float> & fc2_bias;

    const int image_size = 28;
    const int padded_image_size = 30;
    const int input_channel_num = 1;
    const int output_channel_num = 5;
    const int kernel_size = 3;
    const int stride = 1;
    const int paddding = 1;
    constexpr int fc1_hidden_size = 32 * 28 * 28;
    const int fc2_hidden_size = 1024;
};

MnistConv::MnistConv(const std::vector<float> & conv1_weight, const std::vector<float> & conv1_bias,
                     const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias,
                     const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias) :
    : conv1_weight{conv1_weight}, conv1_bias{conv1_bias}, fc1_weight{fc1_weight}, fc1_bias{fc1_bias},
      fc2_weight{fc2_weight}, fc2_bias{fc2_bias} {}

MnistConv::padding(std::vector<float> & data)
{
    std::vector<float> padded_data(padded_image_size * padded_image_size, 0.0f);
    for (int i = 0; i < image_size; i++)
    {
        float * dst = &padded_data[(i+1) * padded_image_size + 1];
        float * src = &data[i * image_size];
        std::memcpy(dst, src, image_size * sizeof(float));
    }
    return padded_data;
}

MnistConv::forward(std::vector<float> & data)
{
    data = padding(data);
    for (auto & d : data)
    {
        std::cout << d << " ";
    }
    return 0;
}

int main(int argc, char * argv[])
{
    MnistConv model(conv1_weight, conv1_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias);
    int out = model.forward(data);
    std::cout << "Prediction: " << out << std::endl;
    return 0;
}
