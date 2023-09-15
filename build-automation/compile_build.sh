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

cmake -B "${build_output}" "${repo_dir}"
cd "${build_output}" && make -j8