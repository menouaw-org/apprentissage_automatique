cmake_minimum_required(VERSION 4.2)

project(pa_ml C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(
        pa_ml SHARED
        src/api/ml_library.c
        src/api/ml_library.h
)

target_include_directories(
        pa_ml PUBLIC
        src/api
)