#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#include "quantized_fc1.h"
#include "quantized_fc2.h"
#include "data_7.h"

struct QuantizedBuffer
{
    std::vector<int8_t> q;
    float s;
};

class MnistFC
{
public:
    MnistFC(const QuantizedBuffer & qfc1, const std::vector<float> & fc1_bias,
            const QuantizedBuffer & qfc2, const std::vector<float> & fc2_bias);

    QuantizedBuffer quantize(const std::vector<float> & data);

    int forward_fp32(const std::vector<float> & data);
    int forward_int8(const std::vector<float> & data);

    void fc1(std::vector<float> & hidden, const std::vector<float> & data);
    void fc1(QuantizedBuffer & hidden, const QuantizedBuffer & data);

    void relu(std::vector<float> & hidden);
    void relu(QuantizedBuffer & relu_hidden, const QuantizedBuffer & hidden);

    void fc2(std::vector<float> & output, const std::vector<float> & hidden);
    void fc2(QuantizedBuffer & output, const QuantizedBuffer & relu_hidden);

public:
    const QuantizedBuffer & qfc1;
    const QuantizedBuffer & qfc2;
    const std::vector<float> & fc1_bias;
    const std::vector<float> & fc2_bias;

    int input_dim = 784;
    int hidden_dim = 128;
    int output_dim = 10;

};

MnistFC::MnistFC(const QuantizedBuffer & qfc1, const std::vector<float> & fc1_bias,
                 const QuantizedBuffer & qfc2, const std::vector<float> & fc2_bias) 
    : qfc1{qfc1}, fc1_bias{fc1_bias}, qfc2{qfc2}, fc2_bias{fc2_bias} {}


int MnistFC::forward_fp32(const std::vector<float> & data)
{
    // convert int8 weight to float32 and calculate for checking
    std::vector<float> hidden(hidden_dim);
    fc1(hidden, data);
    relu(hidden);

    // fc2
    std::vector<float> output(output_dim);
    fc2(output, hidden);

    int max_index = 0;
    float max_value = output[0];
    for (int i = 0; i < output_dim; i++)
    {
        if (output[i] > max_value)
        {
            max_index = i;
            max_value = output[i];
        }
    }
    return max_index;
}

int MnistFC::forward_int8(const std::vector<float> & data)
{
    QuantizedBuffer qdata = quantize(data);
    QuantizedBuffer hidden;
    fc1(hidden, qdata);

    QuantizedBuffer relu_hidden;
    relu(relu_hidden, hidden);

    QuantizedBuffer output;
    fc2(output, relu_hidden);

    int max_index = 0;
    int8_t max_value = output.q[0];
    for (int i = 0; i < output_dim; i++)
    {
        if (output.q[i] > max_value)
        {
            max_index = i;
            max_value = output.q[i];
        }
    }
    return max_index;
}

void MnistFC::relu(QuantizedBuffer & relu_hidden, const QuantizedBuffer & hidden)
{
    std::vector<float> hidden_fp32 (hidden.q.size());
    for (int i = 0; i < hidden_fp32.size(); i++)
    {
        hidden_fp32[i] = std::max(0.0f, static_cast<float>(hidden.q[i]) * hidden.s);
    }

    float min_val = *std::min_element(hidden_fp32.begin(), hidden_fp32.end());
    float max_val = *std::max_element(hidden_fp32.begin(), hidden_fp32.end());

    relu_hidden.s = (max_val - min_val) / 254.0f;
    relu_hidden.q.resize(hidden_fp32.size());
    for (int i = 0; i < relu_hidden.q.size(); i++)
    {
        relu_hidden.q[i] = static_cast<int8_t>(std::clamp(std::round(hidden_fp32[i] / relu_hidden.s), -127.0f, 127.0f));
    }
}

void MnistFC::relu(std::vector<float> & hidden)
{
    for (int i = 0; i < hidden_dim; i++)
    {
        hidden[i] = std::max(0.0f, hidden[i]);
    }
}

void MnistFC::fc1(QuantizedBuffer & hidden, const QuantizedBuffer & data)
{
    /* calculate scale based on W, x */
    float scale = data.s * qfc1.s;

    /* convert bias -> int8 */
    std::vector<int32_t> bias_int32 (fc1_bias.size());
    for (int i = 0; i < fc1_bias.size(); i++)
    {
        bias_int32[i] = static_cast<int32_t>(std::round(fc1_bias[i] / scale));
    }

    /* calculate int8 */
    hidden.q.resize(hidden_dim);
    std::vector<int32_t> hidden_test (hidden_dim);

    for (int i = 0; i < hidden_dim; i++)
    {
        int32_t value = 0;
        for (int j = 0; j < input_dim; j++)
        {
            value += static_cast<int32_t>(qfc1.q[i * input_dim + j]) * static_cast<int32_t>(data.q[j]);
        }
        value = value + bias_int32[i];
        hidden_test[i] = value;
    }
    
    /* requantize the output vector */
    int32_t max_val = *std::max_element(hidden_test.begin(), hidden_test.end());
    int32_t min_val = *std::min_element(hidden_test.begin(), hidden_test.end());
    hidden.s = (max_val - min_val) / 254.0f;
    for (int i = 0; i < hidden_test.size(); i++)
    {
        hidden.q[i] = static_cast<int8_t>(std::clamp(static_cast<float>(hidden_test[i]) / hidden.s, -127.0f, 127.0f));
    }

}

void MnistFC::fc1(std::vector<float> & hidden, const std::vector<float> & data)
{
    std::vector<float> fc1_weight(qfc1.q.size());
    // convert int8 weight into float32
    for (int i = 0; i < fc1_weight.size(); i++)
    {
        fc1_weight[i] = static_cast<float>(qfc1.q[i]) * qfc1.s;
    }

    for (int i = 0; i < hidden_dim; i++)
    {
        float value = 0;
        for (int j = 0; j < input_dim; j++)
        {
            value += fc1_weight[i * input_dim + j] * data[j];
        }
        hidden[i] = value + fc1_bias[i];
    }
}

void MnistFC::fc2(QuantizedBuffer & output, const QuantizedBuffer & relu_hidden)
{
    /* calculate scale based on W, x */
    float scale = relu_hidden.s * qfc2.s;

    /* convert bias -> int8 */
    std::vector<int32_t> bias_int32 (fc2_bias.size());
    for (int i = 0; i < fc2_bias.size(); i++)
    {
        bias_int32[i] = static_cast<int32_t>(std::round(fc2_bias[i] / scale));
    }

    /* calculate int8 */
    output.q.resize(output_dim);
    std::vector<int32_t> output_i32 (output_dim);

    for (int i = 0; i < output_dim; i++)
    {
        int32_t value = 0;
        for (int j = 0; j < hidden_dim; j++)
        {
            value += static_cast<int32_t>(qfc2.q[i * hidden_dim + j]) * static_cast<int32_t>(relu_hidden.q[j]);
        }
        value = value + bias_int32[i];
        output_i32[i] = value;
    }
    
    /* requantize the output vector */
    int32_t max_val = *std::max_element(output_i32.begin(), output_i32.end());
    int32_t min_val = *std::min_element(output_i32.begin(), output_i32.end());
    output.s = (max_val - min_val) / 254.0f;
    for (int i = 0; i < output_i32.size(); i++)
    {
        output.q[i] = static_cast<int8_t>(std::clamp(static_cast<float>(output_i32[i]) / output.s, -127.0f, 127.0f));
    }
}

void MnistFC::fc2(std::vector<float> & output, const std::vector<float> & hidden)
{
    std::vector<float> fc2_weight(qfc2.q.size());
    for (int i = 0; i < fc2_weight.size(); i++)
    {
        fc2_weight[i] = static_cast<float>(qfc2.q[i]) * qfc2.s;
    }

    for (int i = 0; i < output_dim; i++)
    {
        float value = 0;
        for (int j = 0; j < hidden_dim; j++)
        {
            value += fc2_weight[i * hidden_dim + j] * hidden[j];
        }
        output[i] = value + fc2_bias[i];
    }
}

QuantizedBuffer MnistFC::quantize(const std::vector<float> & data)
{
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    float s = (max_val - min_val) / 254.0f;
    
    std::vector<int8_t> quantized (data.size());
    for (int i = 0; i < data.size(); i++)
    {
        float clamped_qw = std::clamp(std::round(data[i] / s), -127.0f, 127.0f);
        quantized[i] = static_cast<int8_t>(clamped_qw);
    }

    return QuantizedBuffer { quantized, s };
}


int main(int argc, char * argv [])
{
    QuantizedBuffer qfc1 { fc1_weight, fc1_scale };
    QuantizedBuffer qfc2 { fc2_weight, fc2_scale };
    MnistFC model(qfc1, fc1_bias, qfc2, fc2_bias);
    int out = model.forward_int8(data);
    std::cout << "Prediction: " << out << std::endl;
    return 0;
}
