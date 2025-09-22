cmake_minimum_required(VERSION 3.10)
project(quantnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(mnist_fc_float32 src/fp32/mnist_fc_fp32.cpp)
add_executable(convert_fp32_int8 src/weight_fp32_to_int8/convert.cpp)
add_executable(inference_int8 src/weight_fp32_to_int8/inference_int8.cpp)
