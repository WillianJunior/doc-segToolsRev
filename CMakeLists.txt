cmake_minimum_required(VERSION 2.8.9)
project (Segmentation)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -ggdb -gdwarf-2")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

set(USE_BASE_SEG ON CACHE BOOL "Compile base serial version.")

if(USE_BASE_SEG)
	add_subdirectory("seg-base")
endif(USE_BASE_SEG)