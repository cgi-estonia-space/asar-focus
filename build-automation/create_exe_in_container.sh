#!/bin/bash

# Simply builds the given CMake project in a container to a directory given as arguments

set -e

if [ "$#" -ne 2 ]; then
    echo "$0 requires 2 arguments."
    echo "Usage:"
    echo "$0 [repo dir where root CMakeLists.txt resides] [docker image]"
    exit 1
fi

# Resolve it to full path to avoid problems later.
repo_dir=$(readlink -m "$1")
docker_image="$2"

# Check if the path is a directory
if [ -d "$repo_dir" ]; then
  # Extract the last folder name
  repo_folder_name=$(basename "$repo_dir")
else
  echo "Given repo dir ${repo_dir} is not a directory or does not exist."
  exit 1
fi

# Check if the image name includes a tag (':' character)
if [[ $docker_image == *":"* ]]; then
  # Extract image name when tag is supplied
  image_name=$(echo "$docker_image" | awk -F '/' '{print $NF}' | awk -F ':' '{print $1}')
else
  # Extract image name when no tag is supplied
  image_name=$(echo "$docker_image" | awk -F '/' '{print $NF}')
fi


container_name="${image_name}-${repo_folder_name}"
container_work_dir="/root/"
docker run -t -d --name "${container_name}" "${docker_image}"
docker cp "${repo_dir}" "${container_name}":"${container_work_dir}/"
container_work_dir_repo="${container_work_dir}/${repo_folder_name}"
echo "container_work_dir_repo - ${container_work_dir_repo}"
if [ -n "$CUDAARCHS" ]; then
  cuda_arch_value="CUDAARCHS=\"$CUDAARCHS\""
fi
docker exec -t "${container_name}" bash -c "$cuda_arch_value ${container_work_dir_repo}/build-automation/compile_build.sh ${container_work_dir_repo} ${container_work_dir_repo}/build"
docker exec -t "${container_name}" bash -c "ln -s ${container_work_dir_repo}/build/asar_focus /usr/bin/asar_focus"
docker stop "${container_name}"
docker commit "${container_name}" "${container_name}-exe"
docker rm "${container_name}"
