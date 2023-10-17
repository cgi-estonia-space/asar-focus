#!/bin/bash

# E2E tests on a prepared container image and Skript files.

set -e

if [ "$#" -lt 2 ]; then
    echo "$0 requires at least 2 arguments."
    echo "Usage:"
    echo "$0 <End-to-end test asset dir> <docker image> [credentials file]"
    exit 1
fi

# Resolve it to full path to avoid problems later.
e2e_dir=$(readlink -m "$1")
docker_image="$2"

# Check if the path is a directory
if [ -d "$e2e_dir" ]; then
  # Extract the last folder name
  e2e_dir_name=$(basename "$e2e_dir")
else
  echo "Given repo dir ${e2e_dir} is not a directory or does not exist."
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

mkdir -p "${e2e_dir}"

container_name="${image_name}-e2e"
container_work_dir="/root/e2e"
set +e
docker stop "${container_name}"
docker rm "${container_name}"
set -e
docker run -t -d --gpus all -v "${e2e_dir}":"${container_work_dir}" --name "${container_name}" "${docker_image}"

if [ "$#" -eq 3 ]; then
  docker cp "$3" "${container_name}":/root/.aws/credentials
fi

docker exec -t "${container_name}" bash -c "cd /alus/skript && python3 skript.py ${container_work_dir}/skripts/*.yaml"
docker stop "${container_name}"
docker rm "${container_name}"
