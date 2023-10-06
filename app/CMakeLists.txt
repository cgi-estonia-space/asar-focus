list(APPEND APP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

list(APPEND APP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/args.cc)

add_library(app-static STATIC ${APP_SOURCES})
target_include_directories(app-static
        PUBLIC
        ${APP_INCLUDE_DIRS}
        )
target_link_libraries(app-static
        PUBLIC
        Boost::program_options
        )
set_target_properties(app-static
        PROPERTIES
        OUTPUT_NAME app
        )

if (ENABLE_TESTS)
    add_subdirectory(unit-test)
endif ()