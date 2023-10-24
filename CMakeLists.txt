cmake_minimum_required(VERSION 3.26)

project(VideoMapper)
set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(VideoMapperLib
        VideoMapper.cpp VideoMapper.h
        )

add_executable(example main.cpp)
target_link_libraries(example VideoMapperLib)
target_link_libraries(example ${OpenCV_LIBS})
