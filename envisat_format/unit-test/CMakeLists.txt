
list(APPEND ENVISAT_FORMAT_UNIT_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/asar_constants_test.cc
        ${CMAKE_CURRENT_LIST_DIR}/envisat_types_test.cc
        ${CMAKE_CURRENT_LIST_DIR}/envisat_parse_util_test.cc
        ${CMAKE_CURRENT_LIST_DIR}/envisat_utils_test.cc
        ${CMAKE_CURRENT_LIST_DIR}/parse_util_test.cc
)

add_executable(envisat-format-unit-test ${ENVISAT_FORMAT_UNIT_TEST_SOURCES})
target_link_libraries(envisat-format-unit-test
        PRIVATE
        gmock_main
        envisat-format-static
)

set_target_properties(envisat-format-unit-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
)
