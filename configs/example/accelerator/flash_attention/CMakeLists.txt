cmake_minimum_required(VERSION 3.20)
project(flash_attention LANGUAGES C CXX ASM)

if(NOT DEFINED ENV{M5_PATH})
    message(FATAL_ERROR "M5_PATH must point at the gem5 root")
endif()
set(M5_PATH $ENV{M5_PATH})

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(BENCH_SRC ${CMAKE_CURRENT_SOURCE_DIR})

set(GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})

add_subdirectory(sw)   # host-side libraries + tooling
add_subdirectory(hw)   # the accelerator (kernels + control/vector firmware)
