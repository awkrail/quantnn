cmake_minimum_required(VERSION 3.10)
project(quantnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# MNIST - MLP
## simple MLP float32
add_executable(mlp_float32 src/mlp/fp32/mnist_fc.cpp)
add_executable(mlp_quantize_weight src/mlp/dynamic_quantization/quantize_weight.cpp)
add_executable(mlp_dynamic_quantization src/mlp/dynamic_quantization/inference.cpp)
add_executable(mlp_calibration src/mlp/static_quantization/calibration.cpp)
add_executable(mlp_static_quantization src/mlp/static_quantization/inference.cpp)

# Mnist - ConvNet float32
add_executable(conv_float32 src/conv/fp32/mnist_conv.cpp)
add_executable(conv_quantize_weight src/conv/dynamic_quantization/quantize_weight.cpp)
add_executable(conv_dynamic_quantization src/conv/dynamic_quantization/inference.cpp)
add_executable(conv_calibration src/conv/static_quantization/calibration.cpp)
add_executable(conv_static_quantization src/conv/static_quantization/inference.cpp)
