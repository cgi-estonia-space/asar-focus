list(APPEND ENVISAT_FORMAT_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

list(APPEND ENVISAT_FORMAT_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/asar_aux.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/asar_constants.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/doris_orbit.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/envisat_aux_file.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/envisat_format_kernels.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/envisat_im_parse.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/envisat_lvl0_parser.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/envisat_lvl1_writer.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/envisat_parse_util.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/ers_im_parse.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/ers_aux_file.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/ers_tim.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/patc.cc
)

add_library(envisat-format-static STATIC ${ENVISAT_FORMAT_SOURCES})
target_include_directories(envisat-format-static
        PUBLIC
        ${ENVISAT_FORMAT_INCLUDE_DIRS}
)
target_link_libraries(envisat-format-static
        PUBLIC
        Boost::date_time
        fmt::fmt
        util-static
)
set_target_properties(envisat-format-static
        PROPERTIES
        OUTPUT_NAME envisat-format
)

if (ALUS_ENABLE_TESTS)
   add_subdirectory(unit-test)
endif ()
