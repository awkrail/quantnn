cmake_minimum_required(VERSION 3.10)
project(quantnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# MNIST - MLP
## simple MLP float32
add_executable(mlp_float32 src/mlp/fp32/mnist_fc.cpp)
add_executable(quantize_weight src/mlp/dynamic_quantization/quantize_weight.cpp)
add_executable(dynamic_quantization src/mlp/dynamic_quantization/inference_int8.cpp)
add_executable(calibration src/mlp/static_quantization/calibration.cpp)
add_executable(static_quantization src/mlp/static_quantization/inference_int8.cpp)

# Mnist - ConvNet float32
add_executable(convnet_float32 src/conv/fp32/mnist_conv.cpp)
add_executable(quantize_weight_conv src/conv/dynamic_quantization/quantize_weight.cpp)
add_executable(conv_dynamic_quantization src/conv/dynamic_quantization/inference.cpp)
