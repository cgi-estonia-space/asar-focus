#!/bin/bash

# Simply builds the given CMake project in a container(s) to a directory given as arguments
set -e

if [ "$#" -lt 3 ]; then
    echo "$0 requires at least 3 arguments."
    echo "Usage:"
    echo "Usage: $0 <repo dir where root CMakeLists.txt resides> <build output dir> <docker_image1> [docker_image2] ..."
    exit 1
fi

# Check if the first two arguments are directories
if [ ! -d "$1" ] || [ ! -d "$2" ]; then
    echo "The first two arguments must be valid directories."
    exit 1
fi

# Iterate through Docker images (starting from the 3rd argument)
for ((i = 3; i <= $#; i++)); do
    docker_image="${!i}"  # Get the argument by index

    # Check if the Docker image is not empty
    if [ -n "$docker_image" ]; then
        echo "Building on Docker image: $docker_image"
        ./compile_build_container.sh "$1" "$2" "${docker_image}"
    fi
done