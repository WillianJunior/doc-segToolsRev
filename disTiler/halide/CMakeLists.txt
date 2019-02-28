cmake_minimum_required(VERSION 2.8.9)
project (disTiler-halide)
add_definitions(-std=c++11)

# add opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add mpi
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# add pthreads
find_package (Threads REQUIRED)

# add halide
find_path(HALIDE_BUILD_PATH REQUIRED)
include_directories(${HALIDE_BUILD_PATH}/include)
link_directories(${HALIDE_BUILD_PATH}/lib)

# create blur jit lib
set(libblurjit libblurjit)
set(libblurjitsrcs BlurJIT.cpp)
add_library("${libblurjit}" STATIC ${libblurjitsrcs})
target_include_directories("${libblurjit}" PUBLIC .)
target_link_libraries(${libblurjit} ${OpenCV_LIBS})

# create blur aot lib
add_executable(libblur ${HALIDE_BUILD_PATH}/../tools/GenGen.cpp BlurAOT.cpp)
# set(libblur libblur)
# set(libblurrcs BlurAOT.cpp ${HALIDE_BUILD_PATH}/../tools/GenGen.cpp)
# add_library("${libblur}" STATIC ${libblurrcs})
# target_include_directories("${libblur}" PUBLIC .)
set_target_properties(libblur PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(libblur PROPERTIES ENABLE_EXPORTS ON)
# target_compile_options(libblur PRIVATE -std=c++11)
target_compile_options(libblur PRIVATE -fno-rtti)
target_link_libraries(libblur Halide)

# create priority queue lib
set(libpq libpq)
set(libpqsrcs PriorityQ.h)
add_library("${libpq}" STATIC ${libpqsrcs})
set_target_properties(libpq PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories("${libpq}" PUBLIC .)
target_link_libraries(${libpq} ${OpenCV_LIBS})

# create distTiling lib
set(libdistiler libdistiler)
set(libdistilersrcs distTiling.cpp)
add_library("${libdistiler}" STATIC ${libdistilersrcs})
target_include_directories("${libdistiler}" PUBLIC .)
target_compile_options(libdistiler PRIVATE -pthread)
target_link_libraries(${libdistiler} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${libdistiler} ${OpenCV_LIBS})
target_link_libraries(${libdistiler} ${MPI_CXX_LIBRARIES})
target_link_libraries(${libdistiler} Halide)
target_link_libraries(${libdistiler} libblur)
target_link_libraries(${libdistiler} libpq)

# create main executable
add_executable(disTiler main.cpp)
target_compile_options(disTiler PRIVATE -std=c++11)
# target_compile_options(disTiler PRIVATE -pthread)
# target_link_libraries (disTiler ${CMAKE_THREAD_LIBS_INIT})
# set_target_properties(disTiler PROPERTIES LINKER_LANGUAGE CXX)
# target_compile_options(disTiler PRIVATE -DUSE_SYSTEM_HDF5=OFF)
target_link_libraries(disTiler ${OpenCV_LIBS})
target_link_libraries(disTiler ${MPI_CXX_LIBRARIES})
target_link_libraries(disTiler Halide)
target_link_libraries(disTiler libblurjit)
target_link_libraries(disTiler libdistiler)

#Copy the test images to the build folder
if(CPY_BASE_SEG_IMG)
	add_custom_target(cpyImagesDTH 
		COMMAND ${CMAKE_COMMAND} -E copy 
		${Segmentation_SOURCE_DIR}/images/* ${CMAKE_CURRENT_BINARY_DIR})
	add_dependencies(disTiler cpyImagesDTH)

	add_custom_target(cpyRunDTH 
		COMMAND ${CMAKE_COMMAND} -E copy 
		${CMAKE_CURRENT_SOURCE_DIR}/run.sh ${CMAKE_CURRENT_BINARY_DIR})
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

