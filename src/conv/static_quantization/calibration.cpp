#include <iostream>
#include <vector>
#include <cstring>

#include "mnist_conv.h"
#include "calibration_data.h"

struct Scale
{
    float input_scale;
    float conv1_scale;
    float fc1_scale;
    float relu_scale;
    float fc2_scale;
};

float calculate_scale_int8(std::vector<std::vector<float>> & data)
{
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
    float scale = std::max(std::abs(max_val), std::abs(min_val)) / 127.0f;
    return scale;
}

float calculate_scale_uint8(std::vector<std::vector<float>> & data)
{
    float max_val = 1e-5;
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[i].size(); j++)
        {
            max_val = std::max(data[i][j], max_val);
        }
    }
    float scale = max_val / 255.0f;
    return scale;
}

class MnistConv
{
public:
    MnistConv(const std::vector<float> & conv1_weight, const std::vector<float> & conv1_bias,
              const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias,
              const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias);

    std::vector<std::vector<float>> padding(std::vector<std::vector<float>> & data);
    std::vector<std::vector<float>> conv1(std::vector<std::vector<float>> & data);
    std::vector<std::vector<float>> fc1(std::vector<std::vector<float>> & data);
    std::vector<std::vector<float>> relu(std::vector<std::vector<float>> & data);
    std::vector<std::vector<float>> fc2(std::vector<std::vector<float>> & data);
    Scale calibrate(std::vector<std::vector<float>> & data);

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
    const int fc1_input_dim = output_channel_num * image_size * image_size;
    const int fc1_hidden_dim = 128;
    const int fc2_hidden_dim = 10;
};

MnistConv::MnistConv(const std::vector<float> & conv1_weight, const std::vector<float> & conv1_bias,
                     const std::vector<float> & fc1_weight, const std::vector<float> & fc1_bias,
                     const std::vector<float> & fc2_weight, const std::vector<float> & fc2_bias)
    : conv1_weight{conv1_weight}, conv1_bias{conv1_bias}, fc1_weight{fc1_weight}, fc1_bias{fc1_bias},
      fc2_weight{fc2_weight}, fc2_bias{fc2_bias} {}

std::vector<std::vector<float>> MnistConv::padding(std::vector<std::vector<float>> & data)
{
    std::vector<std::vector<float>> padded_data_list (data.size());
    for (int i = 0; i < data.size(); i++)
    {
        std::vector<float> padded_data(padded_image_size * padded_image_size, 0.0f);
        for (int j = 0; j < image_size; j++)
        {
            float * dst = &padded_data[(j + pad_size) * padded_image_size + pad_size];
            float * src = &data[i][j * image_size];
            memcpy(dst, src, image_size * sizeof(float));
        }
        padded_data_list[i] = padded_data;
    }
    return padded_data_list;
}

std::vector<std::vector<float>> MnistConv::conv1(std::vector<std::vector<float>> & data)
{
    std::vector<std::vector<float>> conv1_data_list (data.size());
    for (int m = 0; m < data.size(); m++)
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
                            val += data[m][target_index] * conv1_weight[weight_index];
                        }
                    }
                    int output_index = o * oH_size * oW_size + i * oW_size + j;
                    output[output_index] = val + conv1_bias[o];
                }
            }
        }
        conv1_data_list[m] = output;
    }
    return conv1_data_list;
}

std::vector<std::vector<float>> MnistConv::fc1(std::vector<std::vector<float>> & data)
{
    std::vector<std::vector<float>> fc1_output_list (data.size());
    for (int k = 0; k < data.size(); k++)
    {
        std::vector<float> fc1_output (fc1_hidden_dim);
        for (int i = 0; i < fc1_hidden_dim; i++)
        {
            float value = 0.0f;
            for (int j = 0; j < fc1_input_dim; j++)
            {
                value += fc1_weight[i * fc1_input_dim + j] * data[k][j];
            }
            fc1_output[i] = value + fc1_bias[i];
        }
        fc1_output_list[k] = fc1_output;
    }
    return fc1_output_list;
}

std::vector<std::vector<float>> MnistConv::relu(std::vector<std::vector<float>> & data)
{
    for (int k = 0; k < data.size(); k++)
    {
        for (int i = 0; i < fc1_hidden_dim; i++)
        {
            data[k][i] = std::max(0.0f, data[k][i]);
        }
    }
    return data;
}

std::vector<std::vector<float>> MnistConv::fc2(std::vector<std::vector<float>> & data)
{
    std::vector<std::vector<float>> fc2_output_list(data.size());
    for (int k = 0; k < fc2_output_list.size(); k++)
    {
        std::vector<float> fc2_output (fc2_hidden_dim);
        for (int i = 0; i < fc2_hidden_dim; i++)
        {
            float val = 0.0f;
            for (int j = 0; j < fc1_hidden_dim; j++)
            {
                val += fc2_weight[i * fc2_hidden_dim + j] * data[k][j];
            }
            fc2_output[i] = val + fc2_bias[i];
        }
        fc2_output_list[k] = fc2_output;
    }
    return fc2_output_list;
}

Scale MnistConv::calibrate(std::vector<std::vector<float>> & data)
{
    std::vector<std::vector<float>> padded_data = padding(data);
    float input_scale = calculate_scale_int8(padded_data);
    std::vector<std::vector<float>> conv1_data = conv1(padded_data);
    float conv1_scale = calculate_scale_int8(conv1_data);
    std::vector<std::vector<float>> fc1_data = fc1(data);
    float fc1_scale = calculate_scale_int8(fc1_data);
    std::vector<std::vector<float>> relu_data = relu(fc1_data);
    float relu_scale = calculate_scale_uint8(relu_data);
    std::vector<std::vector<float>> fc2_data = fc2(relu_data);
    float fc2_scale = calculate_scale_int8(fc2_data);

    return Scale { input_scale, conv1_scale, fc1_scale, relu_scale, fc2_scale };
}

int main(int argc, char * argv[])
{
    MnistConv model(conv1_weight, conv1_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias);
    Scale scale = model.calibrate(calibration_data);
    
    float input_scale = scale.input_scale;
    float conv1_scale = scale.conv1_scale;
    float fc1_scale = scale.fc1_scale;
    float relu_scale = scale.relu_scale;
    float fc2_scale = scale.fc2_scale;

    std::cout << input_scale << " " << conv1_scale << " " << fc1_scale << " " << relu_scale << " " << fc2_scale << std::endl;

    return 0;
}
