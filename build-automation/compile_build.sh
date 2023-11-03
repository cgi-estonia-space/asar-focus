#!/bin/bash

# Simply builds the given CMake project to a directory given as arguments

set -e

if [ "$#" -ne 2 ]; then
    echo "$0 requires 2 arguments."
    echo "Usage:"
    echo "$0 [repo dir where root CMakeLists.txt resides] [build output dir]"
    exit 1
fi

repo_dir="$1"
build_output="$2"

if [ -n "$CMAKE_BUILD_TYPE" ]; then
  cmake_build_type_option="-DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
fi

cmake "${cmake_build_type_option}" -B "${build_output}" "${repo_dir}"
cd "${build_output}" && make -j8

result=0
if [ -n "$ALUS_ENABLE_TESTS" ]; then
  hardcoded_unit_test_dir="unit-test"

  while read -r exe
  do
    $exe --gtest_output=xml:${hardcoded_unit_test_dir}/results/
    last_result=$?
    result=$((result | last_result))
  done < <(find ${hardcoded_unit_test_dir} -maxdepth 1 -executable -type f)

fi

exit $result
