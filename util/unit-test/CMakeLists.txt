
list(APPEND UTIL_UNIT_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/date_time_util_test.cc
)

add_executable(util-unit-test ${UTIL_UNIT_TEST_SOURCES})
target_link_libraries(util-unit-test
        PRIVATE
        gmock_main
        util-static
)

set_target_properties(util-unit-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
)
