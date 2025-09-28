#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <fstream>

#include "mnist_fc.h"

struct QuantizedBuffer
{
    std::vector<int8_t> q;
    float s;
    uint8_t zero_point = 0;
};


QuantizedBuffer quantize_int8(const std::vector<float> & weight)
{
    float min_val = 1e+5;
    float max_val = 1e-5;
    for (int i = 0; i < weight.size(); i++)
    {
        min_val = std::min(weight[i], min_val);
        max_val = std::max(weight[i], max_val);
    }

    float s = (max_val - min_val) / 127.0f;
    
    std::vector<int8_t> quantized_weight (weight.size());
    for (int i = 0; i < quantized_weight.size(); i++)
    {
        float clamped_qw = std::clamp(std::round(weight[i] / s), -127.0f, 127.0f);
        quantized_weight[i] = static_cast<int8_t>(clamped_qw);
    }

    return QuantizedBuffer { quantized_weight, s, 0 };

}

void dump_as_header_file(const QuantizedBuffer & quantized, const char * prefix, const char * output_file)
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

int main() 
{
    QuantizedBuffer quantized_fc1 = quantize_int8(fc1_weight);
    QuantizedBuffer quantized_fc2 = quantize_int8(fc2_weight);

    dump_as_header_file(quantized_fc1, "fc1", "src/mlp/dynamic_quantization/quantized_fc1.h");
    dump_as_header_file(quantized_fc2, "fc2", "src/mlp/dynamic_quantization/quantized_fc2.h");

    return 0;
}
