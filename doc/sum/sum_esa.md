


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
| CC                   | CUDA Compute Capability             |
| CLI                  | Command-line interface              |
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
datasets/auxiliary files. All the produced files will be written to the location specified by the `output` argument.

If the program has successfully processed the product the return code of the binary execution will be **0**. Any other 
exit code means error during processing which SHALL be accompanied by the message on the console. Unsuccessful
processing could occur due to the invalid input(s), programming error or host machine's/OS errors/congestion.

There is no distinction between ERS and ENVISAT instrument datasets. When there are modes/types given that are not
present for given instrument then this is reported. Same goes for datasets - the processor makes decisions
based on the given input arguments.

## 3.1 CLI arguments

All the processing input is given by the program arguments. No environment variables or configuration files are used.
Somewhat exceptions are mission and instrument specific auxiliary files which are supplied with the directory from
which the processor searches for specific files based on input dataset sensing time. For orbit files either directory or
specific DORIS orbit file path could be given. When not specifying sensing time the processor will process all level 0
input packets if there is sufficient GPU memory.

```
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

As of release version 0.2 only single `type` is supported which is `IMS`. Any other value would result in termination
of processing with accompanying message on console.

### 3.1.1 Hidden CLI features

For debugging and research possibilities couple of 'unofficial' command line arguments are present which are given below.
* `--plot` - create HTML files consisting graphs of the following processing properties - processing velocity,
chirp signal, doppler centroid and scene geolocation properties on earth
* `--intensity` - create GeoTIFF files of the raw measurements, range compressed results and fully compressed intensity

All the created files would be saved in the same directory as the final product given by the `output` argument. They are
prefixed with the level 0 dataset name pattern.

### 3.1.3 Examples

During the production and internal tests there are so called test datasets (TDS) packages used. This is not required
and one could simply have manually assembled the collection of suitable orbit files and auxiliary files downloaded as
a bundle on online at available ESA sites (see [ERS](https://earth.esa.int/eogateway/instruments/sar-ers/auxiliary-data)
and [ENVISAT](https://earth.esa.int/eogateway/instruments/asar/auxiliary-data)).
Both of the scenarios are supported. Below are given examples for both.

**Manually assembled datasets**
```
asar_focus --input SAR_IM__0PWDSI19920904_101949_00000018C087_00294_05950_0000.E1
--aux ers1_aux_dir
--orb ers1_pds_orbit_dir -o /home/foo/ -t IMS
```

**TDS**
```
asar_focus --input
TDS09/ASA_IMS_1PNESA20040229_212506_000000152024_00387_10461_0000.N1/
ASA_IM__0PNPDK20040229_212454_000000492024_00387_10461_2585.N1 
--aux TDS09/ASA_IMS_1PNESA20040229_212506_000000152024_00387_10461_0000.N1
--orb TDS09/ASA_IMS_1PNESA20040229_212506_000000152024_00387_10461_0000.N1/
DOR_VOR_AXVF-P20120424_125300_20040228_215528_20040301_002328
--type IMS --output /home/foo/ --sensing_start 20040229_212504912000
--sensing_stop 20040229_212523002000
```

## 3.2 Input datasets

Single level 0 **envisat format** ERS-1/2 and Envisat mission dataset is required. As of version 0.2 only imaging mode 
datasets are supported. The specification is given in document `PO-RS-MDA-GS-2009 4/C`.

## 3.3 Auxiliary files

As of version 0.2 the following auxiliary files are used:
* DORIS orbit files in envisat format, see [DOR_VOR_AX](https://earth.esa.int/eogateway/catalog/envisat-doris-precise-orbit-state-vectors-dor-vor_ax-)
* Processor configuration file (ASA_CON_AX/ER_CON_AX)
* Instrument characterization file (AUX_INS_AX/ER_INS_AX)
* External calibration data (ASA_XCA_AX/ER_XCA)

For auxiliary file access and documentation please visit [ERS aux](https://earth.esa.int/eogateway/instruments/sar-ers/auxiliary-data)
and [ENVISAT aux](https://earth.esa.int/eogateway/instruments/asar/auxiliary-data) pages at ESA.

# 4 Requirements

Below are specified minimum requirements for hardware. It is based on the resources needed to focus 16 seconds of 
level 0 imaging mode dataset. Resource like CPU is out of scope, because all modern CPUs are suitable. Disk storage
resource is dismal, since only level 1 dataset storage is required also no intermediate files are stored
(except for debugging features discussed in [3.1.1 Hidden CLI features](#311-hidden-cli-features)).

| Property               | Value  |
|------------------------|--------|
| RAM                    | 4 GB   |
| GPU (V)RAM             | 4 GB   |
| NVIDIA CUDA generation | CC 6.0 |

For specific compute capability and NVIDIA GPU models, please see [CUDA wikipedia page](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
