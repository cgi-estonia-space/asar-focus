
list(APPEND APP_UNIT_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/args_test.cc
    )

add_executable(app-unit-test ${APP_UNIT_TEST_SOURCES})
target_link_libraries(app-unit-test
    PRIVATE
    gmock_main
    app-static
    )

set_target_properties(app-unit-test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
    )