set(CMAKE_CUDA_ARCHITECTURES 75)
cmake_minimum_required(VERSION 3.18)
project(cuda_functions CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

set(cuda_ver ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "Cuda Version: ${cuda_ver}")

set(VCPKG_PACKAGE_DIR "$ENV{VCPKG_DIR}\\installed\\x64-windows")

set(CMAKE_PREFIX_PATH "${VCPKG_PACKAGE_DIR}\\share;${CMAKE_PREFIX_PATH}")
message(STATUS "$ Prefix Path: ${CMAKE_PREFIX_PATH}")
find_package(CUDAToolkit REQUIRED)
find_package(LLVM CONFIG REQUIRED)

include_directories("${VCPKG_PACKAGE_DIR}\\include")


add_library(cuda_rtc OBJECT rtc.cpp)
add_library(cuda_event OBJECT event.cpp)
add_library(custring OBJECT custring.cpp)
add_library(custatus OBJECT status.cpp)
target_include_directories(custatus PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cuda_rtc PUBLIC CUDA::cuda_driver CUDA::nvrtc LLVMSupport)
target_link_libraries(custring PUBLIC LLVMSupport)
target_link_libraries(cuda_event PUBLIC CUDA::cuda_driver)
target_compile_options(cuda_event PUBLIC -fcoroutines)

#[[set_target_properties(
        cuda_event
        PROPERTIES
        CXX_STANDARD 20
)]]

#[[if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(LLVMLibPath "$ENV{VCPKG_INSTALLED_PATH}\\debug\\lib")
    set(LLVMLibs LLVMSupportd)
else()
    set(LLVMLibPath "$ENV{VCPKG_INSTALLED_PATH}\\lib")
    set(LLVMLibs LLVMSupport)
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")]]

add_executable(rtc_test rtc_test.cpp $<TARGET_OBJECTS:cuda_rtc> include/util/custring.h)
#target_link_libraries(rtc_test PUBLIC ${LLVMLibs})

add_executable(cuda_functions main.cu $<TARGET_OBJECTS:cuda_event>)
target_link_libraries(cuda_functions PUBLIC CUDA::cuda_driver tbb)

add_executable(test_custring test_custring.cpp $<TARGET_OBJECTS:custring>)
target_link_libraries(test_custring PUBLIC LLVMSupport)
#target_link_libraries(test_custring PUBLIC ${LLVMLibs})

add_executable(test_custatus test_custatus.cpp $<TARGET_OBJECTS:custatus>)



set_target_properties(
        cuda_functions
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)