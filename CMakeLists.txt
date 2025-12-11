cmake_minimum_required(VERSION 3.20)

project(viper)

add_subdirectory(third_party/DiskANN)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(apps)
