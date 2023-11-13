cmake_minimum_required(VERSION 3.20)
project(asar_focus CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SAR_SOURCES
        sar/fractional_doppler_centroid.cu
        sar/fractional_doppler_centroid.cuh
        sar/iq_correction.cu
        sar/iq_correction.cuh
        sar/orbit_state_vector.cpp
        sar/orbit_state_vector.h
        sar/processing_velocity_estimation.cpp
        sar/processing_velocity_estimation.h
        sar/range_compression.cu
        sar/range_compression.cuh
        sar/range_doppler_algorithm.cu
        sar/range_doppler_algorithm.cuh
        sar/sar_chirp.h
        sar/sar_metadata.h
)

set(SOURCES
        main.cc main_flow.cc ${SAR_SOURCES}
)

add_executable(asar_focus ${SOURCES})
set_target_properties(asar_focus PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit 11.2 REQUIRED)
find_package(Boost 1.71.0 REQUIRED COMPONENTS date_time program_options log log_setup)
find_package(GDAL 3.0.4 REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wextra")
# To suppress using of cmath std::min and std::max.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
if (NOT DEFINED ENV{CUDAARCHS})
    message(STATUS "CUDAARCHS not set, defaulting")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "CUDA binary code will be generated for the following architecture(s) - ${CMAKE_CUDA_ARCHITECTURES}")
set_property(TARGET asar_focus PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic ")

INCLUDE_DIRECTORIES(${CUDAToolkit_INCLUDE_DIRS})

if (DEFINED ENV{ALUS_ENABLE_TESTS})
    set(ALUS_ENABLE_TESTS $ENV{ALUS_ENABLE_TESTS})
endif ()

if (ALUS_ENABLE_TESTS)
    message(STATUS "Enabling tests")
    set(ALUS_UNIT_TEST_BINARY_OUTPUT_DIR ${PROJECT_BINARY_DIR}/unit-test)
endif ()

include(FetchContent)
include(dependencies/FetchContents.cmake)

add_subdirectory(app)
add_subdirectory(cuda_util)
add_subdirectory(envisat_format)
add_subdirectory(util)

target_include_directories(asar_focus PRIVATE ${CMAKE_CURRENT_LIST_DIR})
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    message("Also adding 'stdc++fs' for linking since the g++ version ${CMAKE_CXX_COMPILER_VERSION} requires it.")
    target_link_libraries(asar_focus PRIVATE gdal cufft Eigen3::Eigen stdc++fs app-static util-static envisat-format-static cuda-util-static)
else ()
    target_link_libraries(asar_focus PRIVATE gdal cufft Eigen3::Eigen app-static util-static envisat-format-static cuda-util-static)
endif()
