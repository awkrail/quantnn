#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "mnist_fc.h"

class MnistFC
{
public:
    MnistFC(const float * fc1_weight, const float * fc1_bias, 
            const float * fc2_weight, const float * fc2_bias);

public:
    const float * fc1_weight;
    const float * fc1_bias;
    const float * fc2_weight;
    const float * fc2_bias;
    
    int input_dim = 784;
    int hidden_dim = 128;
    int output_dim = 10;

    int fc1_size = input_dim * hidden_dim;
    int fc2_size = hidden_dim * output_dim;
};

MnistFC::MnistFC(const float * fc1_weight, const float * fc1_bias,
                 const float * fc2_weight, const float * fc2_bias) : 
    fc1_weight{fc1_weight}, fc1_bias{fc1_bias}, fc2_weight{fc2_weight}, fc2_bias{fc2_bias} {}

int main(int argc, char * argv[])
{
    MnistFC model(fc1_weight, fc1_bias, fc2_weight, fc2_bias);
    return 0;
}
