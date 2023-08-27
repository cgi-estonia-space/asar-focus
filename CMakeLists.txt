cmake_minimum_required(VERSION 3.20)
project(asar_focus CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

set(SOURCES
        main.cpp
        asar_lvl0_parser.cpp
        asar_constants.cpp
        device_padded_image.cu
        range_doppler_algorithm.cu
        device_padded_image.cuh
        cuda_workplace.h
        cuda_util.h
        range_compression.cu
        cuda_cleanup.h
        checks.h
        cufft_plan.cpp
        cufft_plan.h
        img_output.cpp
        img_output.h
        math_utils.h
        math_utils.cpp
        iq_correction.cu
        iq_correction.cuh
        envisat_file_format.h
        processing_velocity_estimation.h
        processing_velocity_estimation.cpp
        fractional_doppler_centroid.cu
        fractional_doppler_centroid.cuh
        envisat_lvl1_dsd.h
        bswap_util.h
        envisat_sph.h
        envisat_mph.h
        envisat_ph.cpp
        envisat_ph.h
        asar_lvl1_file.cpp
        asar_lvl1_file.h
        envisat_utils.h envisat_ins_file.cpp envisat_ins_file.h orbit_state_vector.cpp orbit_state_vector.h geo_tools.cpp geo_tools.h)

add_executable(asar_focus ${SOURCES})
set_target_properties(asar_focus PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


find_package(CUDAToolkit 11.2 REQUIRED)
find_package(Boost 1.71.0 REQUIRED COMPONENTS date_time)
find_package(GDAL REQUIRED)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wextra")
# To suppress using of cmath std::min and std::max.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
if (NOT DEFINED ENV{CUDAARCHS})
    message(STATUS "CUDAARCHS not set, defaulting")
    set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "CUDA binary code will be generated for the following architecture(s) - ${CMAKE_CUDA_ARCHITECTURES}")
set_property(TARGET asar_focus PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

INCLUDE_DIRECTORIES(${CUDAToolkit_INCLUDE_DIRS})

target_include_directories(asar_focus PRIVATE ${GDAL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    message("Also adding 'stdc++fs' for linking since the g++ version ${CMAKE_CXX_COMPILER_VERSION} requires it.")
    target_link_libraries(asar_focus PRIVATE gdal cufft stdc++fs)
else ()
    target_link_libraries(asar_focus PRIVATE gdal cufft)
endif()

