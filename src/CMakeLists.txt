cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)
set(This cnn)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")
# set(CMAKE_CXX_FLAGS "-O3")

project(${This} CXX C)

include_directories(
    include
    /usr/include
)

add_executable(${PROJECT_NAME} Layer.hpp NeuralNetwork.hpp DenseLayer.hpp nnalgorithms.hpp nnalgorithms.cpp DenseLayer.cpp Layer.cpp NeuralNetwork.cpp main.cpp)

add_subdirectory(DataMining)

target_link_libraries(${PROJECT_NAME} process_power_consumption)
