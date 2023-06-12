cmake_minimum_required(VERSION 3.20)
project(asar_focus CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)



set(SOURCES
        main.cpp
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
        iq_correction.cpp
        iq_correction.h
        iq_correction.cu
        iq_correction.cuh
        envisat_file_format.h
        fractional_doppler_centroid.cu
        fractional_doppler_centroid.cuh
        envisat_dsd.h
        envisat_dsd.cpp
        bswap_util.h
        envisat_sph.h
        envisat_mph.h
        envisat_ph.cpp
        envisat_ph.h
        ers_im_lvl1_file.cpp
        ers_im_lvl1_file.h
        envisat_utils.h)

add_executable(asar_focus ${SOURCES})
set_target_properties(asar_focus PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


find_package(CUDAToolkit 11.2 REQUIRED)

set_property(TARGET asar_focus PROPERTY CUDA_ARCHITECTURES 86)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wextra")
# To suppress using of cmath std::min and std::max.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")


INCLUDE_DIRECTORIES(${CUDAToolkit_INCLUDE_DIRS})


target_link_libraries(asar_focus PRIVATE gdal cufft)
