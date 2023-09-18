#!/bin/bash

# Collects builds/binaries from the container builds out of the given directory
set -e

if [ "$#" -lt 1 ]; then
    echo "$0 requires at least 1 argument."
    echo "Usage:"
    echo "$0 <directory where container builds are> [output dir - optional, default is 1st argument]"
    exit 1
fi

container_build_dir="$1"
output_dir="${container_build_dir}"

if [ "$#" -eq 2 ]; then
  output_dir="$2"
fi

if [ ! -d "${container_build_dir}" ] || [ ! -d "${output_dir}" ]; then
    echo "Not valid directory(ies) supplied."
    exit 1
fi

release_postfix="release"

# Find directories containing executables
directories_with_executables=$(find "$container_build_dir" -type f -executable -name "asar_*" -exec dirname {} \; | sort -u)

# Iterate through directories and create separate tar archives
for dir_with_executables in $directories_with_executables; do
    if [ -d "$dir_with_executables" ]; then
        # Directory name
        dir_name=$(basename "$dir_with_executables")

        # Create a temporary directory
        temp_dir="${output_dir}/${dir_name}_${release_postfix}"
        mkdir -p "${temp_dir}"
        release_archive_name=$(basename "${temp_dir}")

        # Copy executable files matching "asar_*" from the subdirectory to the temporary directory
        find "$dir_with_executables" -type f -executable -name "asar_*" -exec cp {} "$temp_dir" \;

        script_start_dir=$PWD
        cd "$output_dir"
        # Create a tar archive for the subdirectory in the output directory
        tar -czvf "$release_archive_name.tar.gz" "$release_archive_name"

        cd "$script_start_dir"
        # Clean up the temporary directory
        rm -rf "$temp_dir"
    fi
done

