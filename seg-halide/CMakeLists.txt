cmake_minimum_required(VERSION 2.8.9)
project (seg-halide)

# select which halide versions to be compiled
set(USE_HALIDE_SEG_V1 OFF CACHE BOOL "Compile halide parallel version, V1")
if(USE_HALIDE_SEG_V1)
	add_subdirectory("v1")
endif(USE_HALIDE_SEG_V1)
