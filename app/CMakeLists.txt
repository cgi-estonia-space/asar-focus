list(APPEND APP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

list(APPEND APP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/args.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/alus_log.cc)

add_library(app-static STATIC ${APP_SOURCES})
target_include_directories(app-static
        PUBLIC
        ${APP_INCLUDE_DIRS}
        )
target_link_libraries(app-static
        PUBLIC
        Boost::log
        Boost::log_setup
        Boost::program_options
        )
set_target_properties(app-static
        PROPERTIES
        OUTPUT_NAME app
        )

if (ALUS_ENABLE_TESTS)
    add_subdirectory(unit-test)
endif ()