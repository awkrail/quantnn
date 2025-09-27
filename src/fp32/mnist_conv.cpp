#include <iostream>
#include <vector>
#include <string.h>

#include "mnist_conv.h"
#include "data_7.h"


class MnistConv
{
public:
    MnistConv(const std::vector<float> & conv1_weight, const std::vector<float> & conv1_bias,
              const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias,
              const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias);

    std::vector<float> padding(std::vector<float> & data);
    std::vector<float> conv1(std::vector<float> & data);
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
    const int pad_size = 1;
    const int fc1_hidden_size = 5 * 28 * 28;
    const int fc2_hidden_size = 1024;
};

MnistConv::MnistConv(const std::vector<float> & conv1_weight, const std::vector<float> & conv1_bias,
                     const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias,
                     const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias)
    : conv1_weight{conv1_weight}, conv1_bias{conv1_bias}, fc1_weight{fc1_weight}, fc1_bias{fc1_bias},
      fc2_weight{fc2_weight}, fc2_bias{fc2_bias} {}

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

std::vector<float> MnistConv::conv1(std::vector<float> & data)
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
                float val = 0.0f;
                for (int k = 0; k < kernel_size; k++)
                {
                    for (int l = 0; l < kernel_size; l++)
                    {
                        int target_index = (i + k) * padded_image_size + (j + l);
                        int weight_index = o * kernel_size * kernel_size + kernel_size * k + l;
                        val += data[target_index] * conv1_weight[weight_index];
                    }
                }
                int output_index = o * oH_size * oW_size + i * oW_size + j;
                output[output_index] = val + conv1_bias[o];
            }
        }
    }
    return output;
}

int MnistConv::forward(std::vector<float> & data)
{
    data = padding(data);
    data = conv1(data);
    for (auto & d : data)
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    return 0;
}

int main(int argc, char * argv[])
{
    MnistConv model(conv1_weight, conv1_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias);
    int out = model.forward(data);
    std::cout << "Prediction: " << out << std::endl;
    return 0;
}
