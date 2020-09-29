set(CMAKE_CUDA_ARCHITECTURES 75)
cmake_minimum_required(VERSION 3.18)
project(cuda_functions CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_rtc OBJECT rtc.cpp)
add_library(cuda_event OBJECT event.cpp)
target_link_libraries(cuda_rtc PUBLIC CUDA::cuda_driver CUDA::nvrtc LLVMSupport)
target_link_libraries(cuda_event PUBLIC CUDA::cuda_driver)
target_compile_options(cuda_event PUBLIC -fcoroutines)
#[[set_target_properties(
        cuda_event
        PROPERTIES
        CXX_STANDARD 20
)]]

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(LLVMLibPath "$ENV{VCPKG_INSTALLED_PATH}\\debug\\lib")
    set(LLVMLibs LLVMSupportd)
else()
    set(LLVMLibPath "$ENV{VCPKG_INSTALLED_PATH}\\lib")
    set(LLVMLibs LLVMSupport)
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

add_executable(rtc_test rtc_test.cpp $<TARGET_OBJECTS:cuda_rtc>)
target_link_directories(rtc_test PUBLIC ${LLVMLibPath})
target_link_libraries(rtc_test PUBLIC ${LLVMLibs})

add_executable(cuda_functions main.cu $<TARGET_OBJECTS:cuda_event>)
target_link_libraries(cuda_functions PUBLIC CUDA::cuda_driver tbb)

set_target_properties(
        cuda_functions
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)