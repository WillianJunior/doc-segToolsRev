cmake_minimum_required(VERSION 2.8.9)
project (Segmentation)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: Debug Release."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -ggdb -gdwarf-2")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

# base serial version
# set(USE_BASE_SEG ON CACHE BOOL "Compile base serial version.")
# if(USE_BASE_SEG)
add_subdirectory("seg-base")
# endif(USE_BASE_SEG)

set(USE_HALIDE_SEG ON CACHE BOOL "Compile halide parallel version.")
if(USE_HALIDE_SEG)
	add_subdirectory("seg-halide")
endif(USE_HALIDE_SEG)

set(CPY_BASE_SEG_IMG ON CACHE BOOL "Copy example images to serial build dir.")
