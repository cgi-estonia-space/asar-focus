list(APPEND CUDA_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR})

list(APPEND CUDA_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/cuda_device.cu
        ${CMAKE_CURRENT_LIST_DIR}/cuda_device_init.cu
        ${CMAKE_CURRENT_LIST_DIR}/cuda_algorithm.cu
        ${CMAKE_CURRENT_LIST_DIR}/cufft_plan.cc
        ${CMAKE_CURRENT_LIST_DIR}/device_padded_image.cu
        ${CMAKE_CURRENT_LIST_DIR}/memory_policy.cc
)

add_library(cuda-util-static STATIC ${CUDA_UTIL_SOURCES})
target_include_directories(cuda-util-static
        PUBLIC
        ${CUDA_UTIL_INCLUDE_DIRS}
)
target_link_libraries(cuda-util-static
        PUBLIC
        CUDA::cudart
        app-static
)
set_target_properties(cuda-util-static
        PROPERTIES
        OUTPUT_NAME cuda-util
)
