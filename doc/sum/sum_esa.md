


<p style="text-align: center; font-size: xx-large; font-weight: bold">ASAR-FOCUS GPU processor<br>Software User Manual</p>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<p style="text-align: center; font-weight: bold">CGI Estonia<br>Jan 8th 2024<br>Issue/Revision: 1/A</p>
<br>

<p style="text-align: center">Published (including this document) at <a href="https://github.com/cgi-estonia-space/asar-focus">Github<br>https://github.com/cgi-estonia-space/asar-focus</a></p>

<div style="page-break-after: always;"></div>


<h1 style="text-align: center;">CHANGE LOG</h1>

<div align="center">

| Issue/Revision | Date       | Pages | Description                        |
|----------------|------------|-------|------------------------------------|
| 1/A            | 08/01/2024 | All   | Initial document for version 0.2.0 |

</div>

<div style="page-break-after: always;"></div>

<h1 style="text-align: center;">TABLE OF CONTENTS</h1>

<!-- TOC -->
* [1 Introduction](#1-introduction)
* [2 Prerequisites](#2-prerequisites)
  * [2.1 Common components](#21-common-components)
    * [2.1.1 Enabling containers](#211-enabling-containers)
    * [2.1.2 Prebuilt containers](#212-prebuilt-containers)
  * [2.2 Runtime COTS](#22-runtime-cots)
  * [2.3 Development COTS](#23-development-cots)
* [3 Usage](#3-usage)
  * [3.1 CLI arguments](#31-cli-arguments)
  * [3.2 Input datasets](#32-input-datasets)
  * [3.3 Auxiliary files](#33-auxiliary-files)
<!-- TOC -->

<div style="page-break-after: always;"></div>

<h1 style="text-align: center;">DEFINITIONS, ACRONYMS AND ABBREVIATIONS</h1>

<center>

| Acronym/abbreviation | Definition                          |
|----------------------|-------------------------------------|
| COTS                 | Component Off-the-Shelf             |
| CUDA                 | Compute Unified Device Architecture |
| GPU                  | Graphics Processing Unit            |
| RHEL                 | Red Hat Enterprise Linux            |
| SDK                  | Software Development Kit            |

</center>

<div style="page-break-after: always;"></div>

# 1 Introduction

Following information describes the usage of the ERS and Envisat focusser utilizing GPU. It is implemented in C/C++/CUDA
programming languages. Here is described how to install dependencies and build/compile and run the processor. Not all
steps have to be executed in order to run focussing process. There are prebuilt container images and binaries available.
The processor contains **hidden** functionality that could be invoked to
analyze the internals of the processing results. They are described in the chapters below.

Required datasets and auxiliary files are also described. Although for better and more complete overview following
documents are suggested:
* QA4EO-SER-REP-REP-4512
* PO-RS-MDA-GS-2009 Chapter 8

<div style="page-break-after: always;"></div>

# 2 Prerequisites

Currently tested target platforms are Ubuntu 22.04 LTS (codename Jammy and its flavours) and Red Hat Enterprise Linux
version 8 (codename Ootpa). There are 2 set of requirements lists - for development (devel) and for running (runtime).
Common components are NVIDIA GPU driver and CUDA SDK components that should be installed on the processing environment
for both scenarios.

The list of packages and installation scripts are available at [ALUS-platform Github repo](https://github.com/cgi-estonia-space/ALUs-platform).
Based on target platform one can conveniently choose intended scripts. Subchapters consist summary of those.

## 2.1 Common components

| Name          | Version        | Description                                                                                                                                                                         | Instructions                                                                                                                                                                                                                                                                                                                            |
|---------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Nvidia driver | 525 >=         | Component that manages the GPU(s) installed on the system. Also manages low level communication with the device(s). Exposes driver API which is utilized by CUDA Toolkit libraries. | [Ootpa](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/rhel/8/ootpa/install_nvidia_driver.sh)/[Ootpa manual](https://gist.github.com/kautlenbachs/8c6d7966693668d94aa86085403f4821) \| [Jammy](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/ubuntu/22_04/jammy-aws/install_nvidia_container_base.sh)      |
| Cuda SDK      | 11.2 >= 12.0 < | Enables GPU interfacing through the driver API. Also includes GPU tuned libraries for FFT, algebra, etc.                                                                            | [Ootpa](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/rhel/8/ootpa/install_cuda_base.sh)/[Ootpa manual](https://gist.github.com/kautlenbachs/8c6d7966693668d94aa86085403f4821#installation-of-cuda-sdk) \| [Jammy](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/ubuntu/22_04/jammy/install_cuda_base.sh) |                                                                                                                                                                                                                           |


### 2.1.1 Enabling containers

There are many options to choose from which are all summarized on Nvidia's official CDI page at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html.
The scenario could vary from the type of operating system to orchestration framework used. This means there could be 
different engines used - docker, containerd, CRI-O and podman - which all need customized setup. 

Currently utilized and tested approaches within this project are represented below:
* [podman on Ootpa](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/rhel/8/ootpa/install_container_toolkit.sh)
* [docker on Jammy](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/ubuntu/22_04/jammy-aws/install_nvidia_container_base.sh)

### 2.1.2 Prebuilt containers

Below are listed published container images on Dockerhub

| Platform | Development                | Runtime                      |
|----------|----------------------------|------------------------------|
| Ootpa    | `cgialus/alus-ootpa-devel` | `cgialus/alus-ootpa-runtime` |
| Jammy    | `cgialus/alus-devel`       | `cgialus/alus-runtime`       |

## 2.2 Runtime COTS

| Name      | Version         | Description                                             |
|-----------|-----------------|---------------------------------------------------------|
| libstdc++ | == 6.0          | The GNU C++ standard library                            |
| boost     | == 1.71 <= 1.78 | Libraries for CLI arguments, date and time and logging  |
| GDAL      | == 3.0.4        | Creates GeoTIFF files for analyzing purposes            |

Setup scripts:
* [Ootpa](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/rhel/8/ootpa/setup_runtime.sh)
* [Jammy](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/ubuntu/22_04/jammy/setup_runtime.sh)

## 2.3 Development COTS

| Name      | Version         | Description                                             |
|-----------|-----------------|---------------------------------------------------------|
| CMake     | \>= 3.20        | Build files generation and build execution              |
| nvcc      | \>= 11.2 < 12.0 | Compiler for GPU device code                            |
| g++       | \>=8.5 < 12.0   | Compiler for C/C++ host code                            |
| libstdc++ | == 6.0          | The GNU C++ standard library                            |
| fmt       | == 10.1         | String, integer and object text print utility           |
| gtest     | == 1.10         | Automated test development - unit and integration tests |
| boost     | == 1.71 <= 1.78 | Libraries for CLI arguments, date and time and logging  |
| Eigen     | == 3.4          | Algebra library, header only                            |
| GDAL      | == 3.0.4        | Creates GeoTIFF files for analyzing purposes            |

Setup scripts:
* [Ootpa](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/rhel/8/ootpa/setup_devel.sh)
* [Jammy](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/ubuntu/22_04/jammy/setup_dev.sh)

# 3 Usage

The developed processor called `asar_focus` is a single executable GPU ERS and ENVISAT SAR focussing tool. No graphical
user interface or any other hidden requirements are present. Below will be described the arguments, formats and input
datasets/auxiliary files.

## 3.1 CLI arguments

```shell
asar_focus/0.2.0
  -h [ --help ]         Print help
  -i [ --input ] arg    Level 0 ERS or ENVISAT dataset
  --aux arg             Auxiliary files folder path
  --orb arg             Doris Orbit file or folder path
  -o [ --output ] arg   Destination path for focussed dataset
  -t [ --type ] arg     Focussed product type
  --sensing_start arg   Sensing start time in format '%Y%m%d_%H%M%S%F' e.g. 
                        20040229_212504912000. If "sensing_stop" is not 
                        specified, product type based sensing duration is used 
                        or until packets last.
  --sensing_stop arg    Sensing stop time in format '%Y%m%d_%H%M%S%F' e.g. 
                        20040229_212504912000.  Requires "sensing_start" time 
                        to be specified.
  --log arg (=verbose)  Log level, one of the following - 
                        verbose|debug|info|warning|error
```

## 3.2 Input datasets

## 3.3 Auxiliary files