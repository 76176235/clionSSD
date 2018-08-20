cmake_minimum_required(VERSION 3.10)

project(clion_ssd)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(clion_ssd main.cpp utils.cpp objectDetect.cpp)

target_link_libraries(clion_ssd ${OpenCV_LIBS})