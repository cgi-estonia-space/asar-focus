
add_executable(host-memory-allocation ${CMAKE_CURRENT_LIST_DIR}/host_memory_allocation.cc)

target_link_libraries(host-memory-allocation
    PRIVATE
    util-static)

set_target_properties(host-memory-allocation
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_EXPERIMENTS_BINARY_OUTPUT_DIR}
)
