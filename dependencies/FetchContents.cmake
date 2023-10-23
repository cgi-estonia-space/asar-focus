FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.1.1
)
FetchContent_MakeAvailable(fmt)
# https://stackoverflow.com/questions/65860094/how-to-add-eigen-library-to-a-cmake-c-project-via-fetchcontent
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen)

# For Eigen - https://stackoverflow.com/questions/74881296/assert-leads-to-maybe-uninitialized-gcc-warning
# https://github.com/njoy/NJOY21/issues/106
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108230
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
endif ()

if (ALUS_ENABLE_TESTS)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        release-1.10.0
    )
    FetchContent_MakeAvailable(googletest)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(gmock PRIVATE "-Wno-deprecated-copy")
    endif ()
endif ()
