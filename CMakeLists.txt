cmake_minimum_required(VERSION 3.11)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) #WINDOWS

project(
  CMAKETEST
  VERSION 0.1
  LANGUAGES CXX)

#add_subdirectory(lib)
add_executable (CompressionProject main.cpp function.h function.cpp)

target_include_directories(CompressionProject PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(CompressionProject PUBLIC ${CMAKE_SOURCE_DIR}/lib/OpenNI2.lib)
target_link_libraries(CompressionProject PUBLIC ${CMAKE_SOURCE_DIR}/lib/opencv_world480d.lib)

