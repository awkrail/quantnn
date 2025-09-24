cmake_minimum_required(VERSION 3.10)
project(quantnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# simple MLP float32
add_executable(mnist_fc_float32 src/fp32/mnist_fc_fp32.cpp)

# simple MLP int8 dynamic quantization
add_executable(quantize_weight src/dynamic_quantization/quantize_weight.cpp)
add_executable(dynamic_quantization src/dynamic_quantization/inference_int8.cpp)

# simple MLP int8 static quantization
add_executable(calibration src/static_quantization/calibration.cpp)
add_executable(static_quantization src/static_quantization/inference_int8.cpp)
