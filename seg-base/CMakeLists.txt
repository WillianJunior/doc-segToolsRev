cmake_minimum_required(VERSION 2.8.9)
project (seg-serial)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# create serial functions lib
set(libname serialcomponents)
set(libcomponentssrcs ConnComponents.cpp pixelsOps.cpp)
add_library("${libname}" SHARED ${libcomponentssrcs})
target_include_directories("${libname}" PUBLIC .)
target_link_libraries(${libname} ${OpenCV_LIBS})

# create serial test executable
add_executable(segment main.cpp)
target_link_libraries(segment serialcomponents)
target_link_libraries(segment ${OpenCV_LIBS})

#Copy the test images to the build folder
if(CPY_BASE_SEG_IMG)
	add_custom_target(cpyImagesBase COMMAND ${CMAKE_COMMAND} -E copy ${Segmentation_SOURCE_DIR}/images/* ${CMAKE_CURRENT_BINARY_DIR})
	add_dependencies(segment cpyImagesBase)
endif(CPY_BASE_SEG_IMG)
