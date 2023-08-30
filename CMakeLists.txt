cmake_minimum_required(VERSION 3.20)
project(asar_focus CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)


set(ENVISAT_FORMAT_FILES
        envisat_format/asar_constants.cpp
        envisat_format/asar_constants.h
        envisat_format/asar_lvl0_parser.cpp
        envisat_format/asar_lvl0_parser.h
        envisat_format/asar_lvl1_file.cpp
        envisat_format/asar_lvl1_file.h
        envisat_format/bswap_util.h
        envisat_format/envisat_file_format.h
        envisat_format/envisat_aux_file.cpp
        envisat_format/envisat_aux_file.h
        envisat_format/envisat_lvl1_dsd.h
        envisat_format/envisat_mph.h
        envisat_format/envisat_ph.cpp
        envisat_format/envisat_ph.h
        envisat_format/envisat_sph.h
        envisat_format/envisat_types.h
        envisat_format/envisat_utils.h
        envisat_format/doris_orbit.cc
        envisat_format/doris_orbit.h
        )

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

set(CUDA_UTIL_SOURCES
        cuda_util/cuda_cleanup.h
        cuda_util/cuda_util.h
        cuda_util/cuda_workplace.h
        cuda_util/cufft_plan.cpp
        cuda_util/cufft_plan.h
        cuda_util/device_padded_image.cu
        cuda_util/device_padded_image.cuh
        )

set(UTIL_SOURCES
        util/checks.h
        util/geo_tools.cpp
        util/geo_tools.h
        util/img_output.cpp
        util/img_output.h
        util/math_utils.cpp
        util/math_utils.h
        util/plot.cpp
        util/plot.h
        )

set(SOURCES
        main.cpp ${ENVISAT_FORMAT_FILES} ${SAR_SOURCES} ${CUDA_UTIL_SOURCES} ${UTIL_SOURCES}
)

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

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1
)
FetchContent_MakeAvailable(fmt)

target_include_directories(asar_focus PRIVATE ${GDAL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} .)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    message("Also adding 'stdc++fs' for linking since the g++ version ${CMAKE_CXX_COMPILER_VERSION} requires it.")
    target_link_libraries(asar_focus PRIVATE gdal cufft fmt::fmt Boost::date_time stdc++fs)
else ()
    target_link_libraries(asar_focus PRIVATE gdal cufft fmt::fmt Boost::date_time)
endif()

