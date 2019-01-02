cmake_minimum_required(VERSION 2.8.9)
project (seg-halide-v1)

# add opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add halide
find_path(HALIDE_BUILD_PATH REQUIRED)
include_directories(${HALIDE_BUILD_PATH}/include)
link_directories(${HALIDE_BUILD_PATH}/lib)

# # add serial segmentation operations
# include_directories(${seg-serial_SOURCE_DIR})
# link_directories(${seg-serial_SOURCE_DIR})

add_definitions(-std=c++11)
add_executable(halide-v1 main.cpp)
target_link_libraries(halide-v1 Halide)
target_link_libraries(halide-v1 ${OpenCV_LIBS})
target_link_libraries(halide-v1 serialcomponents)

#Copy the test images to the build folder
if(CPY_BASE_SEG_IMG)
	add_custom_target(cpyImagesH1 COMMAND ${CMAKE_COMMAND} -E copy ${Segmentation_SOURCE_DIR}/images/* ${CMAKE_CURRENT_BINARY_DIR})
	add_dependencies(segment cpyImagesH1)
endif(CPY_BASE_SEG_IMG)
