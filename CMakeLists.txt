set(CMAKE_CUDA_ARCHITECTURES 75)
cmake_minimum_required(VERSION 3.18)
project(cuda_functions CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)


add_library(cuda_event OBJECT event.cpp)
target_link_libraries(cuda_event PUBLIC CUDA::cuda_driver)
target_compile_options(cuda_event PUBLIC -fcoroutines)
set_target_properties(
        cuda_event
        PROPERTIES
        CXX_STANDARD 20
)

add_executable(cuda_functions main.cu $<TARGET_OBJECTS:cuda_event>)
target_link_libraries(cuda_functions PUBLIC CUDA::cuda_driver tbb)

set_target_properties(
        cuda_functions
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)