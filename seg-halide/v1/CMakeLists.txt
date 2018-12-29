cmake_minimum_required(VERSION 2.8.9)
project (seg-halide-v1)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


find_path(HALIDE_BUILD_PATH REQUIRED)
include_directories(${HALIDE_BUILD_PATH}/include)
link_directories(${HALIDE_BUILD_PATH}/lib)

add_definitions(-std=c++11)
add_executable(halide-v1 main.cpp)
target_link_libraries(halide-v1 Halide)
