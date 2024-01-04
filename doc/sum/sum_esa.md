


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

| Issue/Revision | Date       | Pages | Description       |
|----------------|------------|-------|-------------------|
| 1/A            | 08/01/2024 | All   | Initial document  |

</div>

<div style="page-break-after: always;"></div>

<h1 style="text-align: center;">Table of contents</h1>

<!-- TOC -->
* [Introduction](#introduction)
<!-- TOC -->

<div style="page-break-after: always;"></div>

<h1 style="text-align: center;">DEFINITIONS, ACRONYMS AND ABBREVIATIONS</h1>

|      |                                     |
|------|-------------------------------------|
| CUDA | Compute Unified Device Architecture |
| GPU  | Graphics Processing Unit            |
| SDK  | Software Development Kit            |
|      |                                     |

<div style="page-break-after: always;"></div>

# Introduction

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

# Prerequisites

Currently tested target platforms are Ubuntu 22.04 LTS (codename Jammy and its flavours) and Red Hat Enterprise Linux
version 8 (codename Ootpa). There are 2 set of requirements lists - for development (devel) and for running (runtime).
Common components are NVIDIA GPU driver and CUDA SDK components that should be installed on the processing environment.

The list of packages and installation scripts are available at [ALUS-platform Github repo](https://github.com/cgi-estonia-space/ALUs-platform)
 based on target platform one can conveniently choose intended scripts. Below is a summary of those.

## Common components

| Name          | Version  | Description                            | Instructions                                                                                                                                                                                                                                                                                                                       |
|---------------|----------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Nvidia driver | \>= 525  | GPU driver and runtime                 | [Ootpa](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/rhel/8/ootpa/install_nvidia_driver.sh)/[Ootpa manual](https://gist.github.com/kautlenbachs/8c6d7966693668d94aa86085403f4821) \| [Jammy](https://github.com/cgi-estonia-space/ALUs-platform/blob/main/ubuntu/22_04/jammy-aws/install_nvidia_container_base.sh) |
| Cuda SDK      | \>= 11.7 | CUDA runtime and development libraries |                                                                                                                                                                                                                                                                                                                                    |




### Enabling containers

There are many options, please see NVIDIA's official guide blaa, but docker and podman are here