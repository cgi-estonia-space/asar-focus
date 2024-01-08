


<p style="text-align: center; font-size: xx-large; font-weight: bold">ASAR-FOCUS GPU processor<br>Architecture Design Document</p>
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
* [1.1 Purpose](#11-purpose)
* [1.2 Scope](#12-scope)
* [2 System Overview](#2-system-overview)
* [3 System Context](#3-system-context)
* [4 System Design](#4-system-design)
* [5 Component Description](#5-component-description)
  * [5.1 DevicePaddedImage](#51-devicepaddedimage)
* [6 Feasability and Resource Estimates](#6-feasability-and-resource-estimates)
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
| OS                   | Operating system                    |
| RHEL                 | Red Hat Enterprise Linux            |
| SDK                  | Software Development Kit            |
| SNAP                 | Sentinel Application Platform       |

</center>

<div style="page-break-after: always;"></div>

# 1 Introduction

Based on http://microelectronics.esa.int/vhdl/pss/PSS-05-04.pdf Chapter 5

# 1.1 Purpose

Using GPUs for earth observation (EO) data processing is a rather undeveloped method. At the same time this is not a
novel way, because such approaches have been developed for more than 10 years. This was enabled by the
NVIDIA's CUDA architecture which makes possible to process generic payloads besides graphical textures. The `asar_focus`
processor is one of the attempts to reproduce the data quality of a legacy CPU based processor while enabling
processing performance gains.

This document aims to facilitate:
* further development of the focusser
* document the principles of efficient EO data processing using parallel processing paradigm


# 1.2 Scope

There is a single executable binary produced by the software package. It is built for Red Hat Enterprise Linux version 
8.x. All the other components required are detailed in the [software user manual](https://github.com/cgi-estonia-space/asar-focus/blob/documentation/doc/sum/sum.md).
The project could be run on other Linux distributions as well, with possible tweaking of the required components.

Current state of the focusser will be able to focus ERS-1/2 and Envisat's synthetic aperture radar (SAR) instrument's 
data for the imaging mode (IM) datasets only. It lacks some functionality in order to match the baseline PF-ASAR
processor functionality:
* PATC/PATN support for ERS time synchronization is missing
* Sections of metadata is not appropriately formed (enough for products to be further processed in ESA SNAP)
* Focussing quality is not matched (visually good, but histograms of focussed I/Q raster are not so well-formed)

Current state of the developed functionality enables to:
* Revisit details of the missions' instrument processing and document it properly for further developments
* Implement all the other acquisition modes' (wave, alternating polarisation etc.) 
  parallel processing pipelines with less effort
* Re-analyze baseline processor PF-ASAR data quality and develop further enhancements
* Lower the costs of the instrument data processing facility by faster and more energy efficient processing

# 2 System Overview

The GPU accelerated processors greatly benefit from the parallel computing paradigm. Previous ESA activities developed
by CGI Estonia together with CGI Italy and industry experts have produced set of GPU accelerated processors available
on [Github](https://github.com/cgi-estonia-space/ALUs). The processing speed and data quality numbers are given in the
repository and its [Wiki](https://github.com/cgi-estonia-space/ALUs/wiki). As a general outcome the processing speed 
could be increased from 10 to 100 times while enabling up to 10 times fewer costs than utilising traditional processors
tha leverage CPU for computing.

In order to achieve fast processing utilising GPU one has to focus on:
* Minimising data transfers between storage, RAM and GPU
* Utilise as fast as possible disk or RAM storage mechanisms
* Utilise GPU as much and often as possible during the execution of the processor

# 3 System Context

This processor's general system has been kept simple which has been validated by the
[ALUs](https://github.com/cgi-estonia-space/ALUs) processors. This has proven to enable further integration with other
systems with possibility to add convenient functionality like graphical user interface or data fetching. Below are
general blocks that define the approach.

![asar_focus system context](https://github.com/kautlenbachs/bulpp_diagrams/blob/main/asar-focus-system-context-a.drawio.png?raw=true)

*Figure 3.0 - General system block scheme*

The OS here facilitates:
* Execution of binary on a system with the GPU
* File input/output
* Memory management
* CPU threading

The CUDA component enables:
* Query and check the GPU properties
* Command GPU
* Transfer data between GPU and RAM

The `asar_focus` processor:
* Command line arguments handling
* Dataset parsing and results formating
* Focussing pipeline implementation
* Efficient disk storage read and write strategies
* Efficient GPU occupation strategies
* System logs
* Error checks for dataset and GPU

# 4 System Design

Main focus is on the efficient GPU processing pipeline where the time spent on I/O should be minimized or parallelized
while processing calculations. Below is the general pipeline of the processor.

![general processing pipeline](https://github.com/kautlenbachs/bulpp_diagrams/blob/main/asar_meta_flow.drawio.png?raw=true)

*Figure 4.0 - General processing pipeline*

The **measurements processing** step is generic focussing pipeline that is represented by the figure below.
![focussing processing pipeline](https://github.com/kautlenbachs/bulpp_diagrams/blob/main/sar_focussing.drawio.png?raw=true)

*Figure 4.1 - Generic focussing pipeline*

Both of these figures above are tightly coupled. As can be seen on the last figure the "raw file decoding" and 
"processed file" are the steps that are mostly presented on the first figure. Based on current profiling results dataset
parsing and writing takes 60 and more percent of the overall end to end time, depending heavily on the storage device
capability.

# 5 Component Description

## 5.1 DevicePaddedImage

SAR focusing requires forward and backwards Fourier transforms in range and azimuth directions, additionally the SAR 
datasets are quite large

DevicePaddedImage is the core abstraction to facilitate efficient SAR transformations by embedding extra padding areas 
in the right and bottom of the image. This allows inplace FFTs in both range and azimuth direction, this alleviates 
the need for extra memory copies to fulfill FFT size requirements. Thus changing from time domain to frequency domain 
and back is done memory efficiently, without needlessly copying multi gigabyte signal data.

Additional design consideration is the fact that GPU onboard memory is less than main board RAM, thus using this
technique allows to keep the full DSP chain on the GPU memory, avoiding costly CPU <-> GPU transfers.
Visual representation of the memory layout.
```
    RANGE
   |------------------------|---------|
 A |                        |         |
 Z | SAR SIGNAL DATA        |  RANGE  |
 I |                        |   FFT   |
 M |                        | PADDING |
 U |                        |         |
 T |------------------------|---------|
 H | AZIMUTH FFT PADDING    |         |
   |------------------------|---------|
```

## 5.2 Envisat format module

The Envisat format support handling and support was implemented during this project. No libraries were used. This includes
custom structures, little/big-endian conversions and data type handling. All the implementation is contained in 
`envisat_format` folder of the project. It applies to both the ERS and Envisat mission datasets

It contains the following sub-functionality:
* DORIS orbit file search and parsing
* Auxiliary files' search and parsing
* Level 0 imaging mode dataset parsing
* Level 1 image mode single look complex dataset structure and writing
* GPU computation kernels to condition the measurements

## 5.3 Boost log

Boost library's submodule implemented in C++ to enable console log handling. This enables structured logs with timestamps
and different log levels. 

An extract of the curren log:
```
[2024-01-08 11:43:57.852098] [0x00007fa101e9b000] [debug]   Doppler centroid constant term -568.25 and max aperture pixels 543
[2024-01-08 11:43:57.852121] [0x00007fa101e9b000] [info]    Need to remove 543 lines before requested sensing stop
[2024-01-08 11:43:57.853812] [0x00007fa101e9b000] [debug]   Azimuth compression time = 536 ms
[2024-01-08 11:43:57.854107] [0x00007fa101e9b000] [trace]   SWST change at line 19814 from 2024 to 2048
[2024-01-08 11:43:57.854160] [0x00007fa101e9b000] [debug]   Swath = IS2 idx = 1
```

## 5.4 Boost program arguments

Boost library's submodule implemented in C++. Used to facilitate command line arguments handling. All the setup, help
messages, validation and parsing is achieved using this module.

Example of the interface built using the module.

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

## 5.4 cuFFT

Submodule of the umbrella CUDA SDK. While other functionality of the CUDA is rather generic and detailed on the NVIDIA
developer sites, the usage of cuFFT is detailed here, because this enables the easier handling of the focussing pipeline.




# 6 Feasability and Resource Estimates

#Profiling analyze

Example GPU profiling chart:

##Test setup:
RTX3060 Laptop GPU<br>
AMD Ryzen 7 5800H CPU<br>
Fast NVMe SSD<br>

##Test data:

Envisat IM(LVL0) -> Envisat IMS(LVL1) with about 16 seconds of signal data

The whole chain is procced in less than 2 seconds

1. File read, metadata parsing & signal data DISK -> CPU -> GPU (350 ms)
2. Core SAR DSP chain (700 ms)
3. Metadata assembly, file writing GPU -> CPU -> DISK (750 ms)

Steps 1 & 3 are heavily dependent on the performance of the storage and can be a major bottleneck.

Step 2 is mostly done the GPU and the most time-consuming steps are the FFTs calculated by cuFFT

The cpu speed should not be relevant, as no major focussing steps are bottleneck by the cpu.

#GPU and RAM memory limit based on the dataset size

The baseline GPU memory requirement is estimated to be:

total memory usage = range size * azimuth size * 8 * 1.3 * 2

range size = lvl0 range size
azimuth size = total number of processed packets
8 => size of complex float
1.3 => rough estimate of the paddings required for range and azimuth compression and additional metadata
2 => memory used for FFTs and output image generation

On a standard 16s IM dataset, this works out to be about 3GB of GPU memory.

CPU memory usage should be roughly the input file size + output file size. Other elements take insignificant amount of memory.


