list(APPEND UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

list(APPEND UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/date_time_util.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/file_async.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/filesystem_util.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/geo_tools.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/img_output.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/math_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/plot.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/globe_plot.cc
        )

add_library(util-static STATIC ${UTIL_SOURCES})
target_include_directories(util-static
        PUBLIC
        ${UTIL_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}/../
)
target_link_libraries(util-static
        PUBLIC
        CUDA::cudart
        gdal
        fmt::fmt
        Boost::date_time
        Eigen3::Eigen
        app-static
        cuda-util-static
)
set_target_properties(util-static
        PROPERTIES
        OUTPUT_NAME util
)

if (ALUS_ENABLE_TESTS)
    add_subdirectory(unit-test)
endif ()
