cmake_minimum_required(VERSION 2.8.9)
project (disTiler-halide)

# add opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add mpi
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# add halide
find_path(HALIDE_BUILD_PATH REQUIRED)
include_directories(${HALIDE_BUILD_PATH}/include)
link_directories(${HALIDE_BUILD_PATH}/lib)

# create blur jit lib
set(libblurjit libblurjit)
set(libblurjitsrcs BlurJIT.cpp)
add_library("${libblurjit}" SHARED ${libblurjitsrcs})
target_include_directories("${libblurjit}" PUBLIC .)
target_link_libraries(${libblurjit} ${OpenCV_LIBS})

# # create blur aot lib
# set(libblurjit libblurjit)
# set(libblurjitsrcs BlurJIT.cpp)
# add_library("${libblurjit}" SHARED ${libblurjitsrcs})
# target_include_directories("${libblurjit}" PUBLIC .)
# target_link_libraries(${libblurjit} ${OpenCV_LIBS})

# create priority queue lib

# create distTiling lib
set(libdistiler libdistiler)
set(libdistilersrcs distTiling.cpp)
add_library("${libdistiler}" SHARED ${libdistilersrcs})
target_include_directories("${libdistiler}" PUBLIC .)
target_link_libraries(${libdistiler} ${OpenCV_LIBS})
target_link_libraries(${libdistiler} ${MPI_C_LIBRARIES})
target_link_libraries(${libdistiler} Halide)

# create main executable
add_definitions(-std=c++11)
add_executable(disTiler main.cpp)
target_link_libraries(disTiler ${OpenCV_LIBS})
target_link_libraries(disTiler ${MPI_C_LIBRARIES})
target_link_libraries(disTiler Halide)
target_link_libraries(disTiler libblurjit)
target_link_libraries(disTiler libdistiler)

#Copy the test images to the build folder
if(CPY_BASE_SEG_IMG)
	add_custom_target(cpyImagesDTH COMMAND ${CMAKE_COMMAND} -E copy ${Segmentation_SOURCE_DIR}/images/* ${CMAKE_CURRENT_BINARY_DIR})
	add_dependencies(disTiler cpyImagesDTH)
	add_custom_target(cpyRunDTH COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/run.sh ${CMAKE_CURRENT_BINARY_DIR})
	add_dependencies(disTiler cpyRunDTH)
endif(CPY_BASE_SEG_IMG)

# # run command to make testing easy
# set(IN in.tiff)
# add_custom_command(TARGET disTiler
# 	POST_BUILD
# 	COMMAND ./disTiler -i $IN -d 0
# 	COMMAND xdg-open $IN 
# 	COMMAND xdg-open output.png 
# 	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# 	VERBATIM
# )

