cmake_minimum_required(VERSION 3.20)
project(asar_focus CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ENVISAT_FORMAT_FILES
        envisat_format/asar_aux.cc
        envisat_format/asar_aux.h
        envisat_format/asar_constants.cpp
        envisat_format/asar_constants.h
        envisat_format/envisat_lvl0_parser.cpp
        envisat_format/envisat_lvl0_parser.h
        envisat_format/envisat_lvl1_writer.cpp
        envisat_format/envisat_lvl1_writer.h
        envisat_format/bswap_util.h
        envisat_format/envisat_mph_sph_parser.h
        envisat_format/envisat_aux_file.cpp
        envisat_format/envisat_aux_file.h
        envisat_format/envisat_lvl1_ads.h
        envisat_format/envisat_lvl1_mph.h
        envisat_format/envisat_mph_sph_str_utils.h
        envisat_format/envisat_lvl1_sph.h
        envisat_format/envisat_types.h
        envisat_format/envisat_utils.h
        envisat_format/ers_aux_file.cc
        envisat_format/ers_aux_file.h
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
        util/filesystem_util.cc
        util/date_time_util.cc
        util/date_time_util.h

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
        main.cc ${ENVISAT_FORMAT_FILES} ${SAR_SOURCES} ${CUDA_UTIL_SOURCES} ${UTIL_SOURCES}
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
    set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "CUDA binary code will be generated for the following architecture(s) - ${CMAKE_CUDA_ARCHITECTURES}")
set_property(TARGET asar_focus PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic")

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

target_include_directories(asar_focus PRIVATE ${CMAKE_CURRENT_LIST_DIR})
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    message("Also adding 'stdc++fs' for linking since the g++ version ${CMAKE_CXX_COMPILER_VERSION} requires it.")
    target_link_libraries(asar_focus PRIVATE gdal cufft fmt::fmt Boost::date_time Boost::program_options Eigen3::Eigen stdc++fs app-static)
else ()
    target_link_libraries(asar_focus PRIVATE gdal cufft fmt::fmt Boost::date_time Boost::program_options Eigen3::Eigen app-static)
endif()
