cmake_minimum_required(VERSION 3.10)
project(quantnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB CORE_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(quantnn
    ${CORE_FILES}
)

target_include_directories(quantnn
    PRIVATE
    include
)

target_compile_options(quantnn
    PRIVATE
    -O0
)
