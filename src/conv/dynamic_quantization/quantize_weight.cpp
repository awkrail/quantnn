#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <fstream>

#include "mnist_conv.h"

template<typename T>
struct QuantizedConvBuffer
{
    std::vector<T> q;
    std::vector<float> s;
};

template<typename T>
struct QuantizedBuffer
{
    std::vector<T> q;
    float s;
};

struct ConvParams
{
    int output_channel_num;
    int kernel_size;
    int stride;
    int pad_size;
};

QuantizedConvBuffer<int8_t> quantize_channel_int8(const std::vector<float> & weight, const ConvParams & conv_params)
{
    std::vector<int8_t> quantized_weight (weight.size());
    std::vector<float> scales (conv_params.output_channel_num);
    for (int i = 0; i < conv_params.output_channel_num; i++)
    {
        int elem_num = conv_params.kernel_size * conv_params.kernel_size;
        int start_index = i * elem_num;
        int end_index = (i+1) * elem_num;

        float max_val = 1e-5;
        float min_val = 1e+5;
        for (int j = start_index; j < end_index; j++)
        {
            max_val = std::max(weight[j], max_val);
            min_val = std::min(weight[j], min_val);
        }
        float scale = std::max(std::abs(max_val), std::abs(min_val)) / 127.0f;
        for (int j = start_index; j < end_index; j++)
        {
            float qval = std::clamp(std::round(weight[j] / scale), -127.0f, 127.0f);
            quantized_weight[j] = static_cast<int8_t>(qval);
        }
        scales[i] = scale;
    }
    return QuantizedConvBuffer<int8_t> { quantized_weight, scales };
}

QuantizedBuffer<int8_t> quantize_int8(const std::vector<float> & weight)
{
    float min_val = 1e+5;
    float max_val = 1e-5;
    for (int i = 0; i < weight.size(); i++)
    {
        min_val = std::min(weight[i], min_val);
        max_val = std::max(weight[i], max_val);
    }
    float scale = std::max(std::abs(max_val), std::abs(min_val)) / 127.0f;
    std::vector<int8_t> quantized_weight (weight.size());
    for (int i = 0; i < quantized_weight.size(); i++)
    {
        float qval = std::clamp(std::round(weight[i] / scale), -127.0f, 127.0f);
        quantized_weight[i] = static_cast<int8_t>(qval);
    }

    return QuantizedBuffer<int8_t> { quantized_weight, scale };
}

void dump_as_header_file(const QuantizedBuffer<int8_t> & quantized, const char * prefix, const char * output_file)
{
    std::ostringstream oss;
    oss << "const std::vector<int8_t> " << prefix << "_weight = { ";
    for (int i = 0; i < quantized.q.size(); i++)
    {
        oss << static_cast<int>(quantized.q[i]);
        if (i + 1 < quantized.q.size())
        {
            oss << ", ";
        }
    }

    oss << " };\n\nconst float " << prefix << "_scale = " << quantized.s << ";\n";
    std::ofstream ofs(output_file);
    ofs << oss.str();
    ofs.close();
}

void dump_as_header_file_conv(const QuantizedConvBuffer<int8_t> & quantized, const char * prefix, const char * output_file)
{
    std::ostringstream oss;
    oss << "const std::vector<int8_t> " << prefix << "_weight = { ";
    for (int i = 0; i < quantized.q.size(); i++)
    {
        oss << static_cast<int>(quantized.q[i]);
        if (i + 1 < quantized.q.size())
        {
            oss << ", ";
        }
    }

    oss << " };\n\nconst std::vector<float> " << prefix << "_scale = { ";
    for (int i = 0; i < quantized.s.size(); i++)
    {
        oss << quantized.s[i] << ", ";
    }
    oss << " };\n";
    std::ofstream ofs(output_file);
    ofs << oss.str();
    ofs.close();
}

int main()
{
    ConvParams conv1_params { 5, 3, 1, 1 };
    QuantizedConvBuffer<int8_t> quantized_conv1 = quantize_channel_int8(conv1_weight, conv1_params);
    QuantizedBuffer<int8_t> quantized_fc1 = quantize_int8(fc1_weight);
    QuantizedBuffer<int8_t> quantized_fc2 = quantize_int8(fc2_weight);

    dump_as_header_file_conv(quantized_conv1, "qconv1", "src/conv/dynamic_quantization/quantized_conv1.h");
    dump_as_header_file(quantized_fc1, "qfc1", "src/conv/dynamic_quantization/quantized_fc1.h");
    dump_as_header_file(quantized_fc2, "qfc2", "src/conv/dynamic_quantization/quantized_fc2.h");

    return 0;
}
