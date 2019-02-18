cmake_minimum_required(VERSION 2.8.9)
project (disTiler)

# select which halide versions to be compiled
set(USE_DISTILER_HALIDE OFF CACHE BOOL "Compile disTiler with halide.")
if(USE_DISTILER_HALIDE)
	add_subdirectory("halide")
endif(USE_DISTILER_HALIDE)
