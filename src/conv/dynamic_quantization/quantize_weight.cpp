#include <vector>

#include "mnist_conv.h"

template<typename T>
struct QuantizedConvBuffer
{
    std::vector<T> q;
    std::vector<float> s;
};


struct ConvParams
{
    int output_channel_num;
    int kernel_size;
    int stride;
    int pad_size;
};


QuantizedConvBuffer quantized_conv1(const std::vector<float> & weight, const ConvParams & conv_params)
{
    std::vector<int8_t> quantized_weight (weight.size());
    std::vector<float> scales (conv_params.output_channel_num);
    for (int i = 0; i < conv_params.output_channel_num; i++)
    {
        int elem_num = conv_params.kernel_size * conv_params.kernel_size;
        int start_index = i * elem_num;
        int end_index = (i+1) * elem_num;

        float min_val = 1e+5;
        float max_val = 1e-5;
        for (int j = start_index; j < end_index; j++)
        {
            min_val = std::min(weight[j], min_val);
            max_val = std::max(weight[j], max_val);
        }

        float scale = (max_val - min_val) / 127.0f;
        for (int j = start_index; j < end_index; j++)
        {
            float qval = std::clamp(std::round(weight[i] / scale), -127.0f, 127.0f);
            quantized_weight[j] = static_cast<int8_t>(qval);
        }
        scales[i] = scale;
    }
    return QuantizedConvBuffer { quantized_weight, scales };
}


int main()
{
    ConvParams conv1_params { 5, 3, 1, 1 };
    QuantizedConvBuffer<int8_t> quantized_conv1 = quantize_channel_int8(conv1_weight, conv1_params);
}
