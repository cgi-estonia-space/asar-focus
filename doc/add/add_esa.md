


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
<p style="text-align: center; font-weight: bold">CGI Estonia<br>Jan 8th 2024<br>Issue/Revision: 1/B</p>
<br>

<p style="text-align: center">Published (including this document) at <a href="https://github.com/cgi-estonia-space/asar-focus">Github<br>https://github.com/cgi-estonia-space/asar-focus</a></p>

<div style="page-break-after: always;"></div>


<h1 style="text-align: center;">CHANGE LOG</h1>

<div align="center">

| Issue/Revision | Date       | Pages/Paragraphs | Description                         |
|----------------|------------|------------------|-------------------------------------|
| 1/A            | 08/01/2024 | All              | Initial document for version 0.2.0  |
| 1/B            | 19/03/2024 | 1.2, 5.2         | Version 0.3.0 supports IMP and PATC |

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
  * [5.2 Envisat format module](#52-envisat-format-module)
  * [5.3 Boost log](#53-boost-log)
  * [5.4 Boost program arguments](#54-boost-program-arguments)
  * [5.4 cuFFT](#54-cufft)
* [6 Feasability and Resource Estimates](#6-feasability-and-resource-estimates)
  * [6.1 Profiling analyze](#61-profiling-analyze)
  * [6.2 GPU and RAM memory limit based on the dataset size](#62-gpu-and-ram-memory-limit-based-on-the-dataset-size)
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

![asar_focus system context](https://private-user-images.githubusercontent.com/98521380/379805603-d07659d5-48f7-4fbf-9661-ba81db2a49ba.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjk3NzkzMDIsIm5iZiI6MTcyOTc3OTAwMiwicGF0aCI6Ii85ODUyMTM4MC8zNzk4MDU2MDMtZDA3NjU5ZDUtNDhmNy00ZmJmLTk2NjEtYmE4MWRiMmE0OWJhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDI0VDE0MTAwMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUzMTIzOTNmNDFlZTkwZjcxODFjNzAwZDkyZTM1ODAzYzdjMGU5NGYwZjNlZDVkNWVlODNjZWQ5MmZmYTEzZmYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.IFwEzsQBKHy3lNfmevUhQI7lUfCxDs-p9v3Nr3gxUG8)

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

![general processing pipeline](https://private-user-images.githubusercontent.com/98521380/380160723-e622e536-894c-4670-8e7a-16be2fe54ca5.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjk4NTk4MDEsIm5iZiI6MTcyOTg1OTUwMSwicGF0aCI6Ii85ODUyMTM4MC8zODAxNjA3MjMtZTYyMmU1MzYtODk0Yy00NjcwLThlN2EtMTZiZTJmZTU0Y2E1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDI1VDEyMzE0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTljMGRiYmZjNzliMTBlZGQ0MWVlN2FiNmM4YTUwNDEyZGY1N2E5NmIwNzllNjYwYWZjYzQ3MDMzYzZhYTExNDcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Yq_JobrN-bzrbgXMB58ZI_hW7vDr9jo_LF7MyROJ1w4)

*Figure 4.0 - General processing pipeline*

The **measurements processing** step is generic focussing pipeline that is represented by the figure below.

![focussing processing pipeline](https://private-user-images.githubusercontent.com/98521380/380164487-f083e9b2-ef14-40b7-825a-3bd5891b1a6a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjk4NjAzODgsIm5iZiI6MTcyOTg2MDA4OCwicGF0aCI6Ii85ODUyMTM4MC8zODAxNjQ0ODctZjA4M2U5YjItZWYxNC00MGI3LTgyNWEtM2JkNTg5MWIxYTZhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDI1VDEyNDEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWIxZWRjMmIyYTNkZjE5NzBkZjM0NDI3OWE5Y2MxYzk2ODgwYzZiNTUyMWU1NDkwNmFlODhlMzAwZmY1OGVjM2ImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Mn8HRrhAYwnSF04WCOMTH1qqCOvxJJa4bmeiWjgH1zI)

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
* Level 1 image mode single look complex and ground range precision datasets structure and writing
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
developer sites, the usage of cuFFT is documented here. 

The SAR focussing pipeline uses the Fourier transforms in range and azimuth direction to generate the final focussed
product. This library can highly perform massively parallelized FFT transforms of the whole signal data in one function
call.


# 6 Feasability and Resource Estimates

## 6.1 Profiling analyze

Example GPU profiling chart is presented below
![profiling chart](https://private-user-images.githubusercontent.com/98521380/380165156-46fad159-30ce-458c-9cea-55dad33d99ef.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjk4NjA0ODIsIm5iZiI6MTcyOTg2MDE4MiwicGF0aCI6Ii85ODUyMTM4MC8zODAxNjUxNTYtNDZmYWQxNTktMzBjZS00NThjLTljZWEtNTVkYWQzM2Q5OWVmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDI1VDEyNDMwMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTFiYjFiZWE4OGVhMWU5NTk5MzNjYWZlM2Q0YmFhNjQxNGNjZGMyMjQ4ODRhNzQ0YTgyOGFlY2E2MjRhMzkwNjImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.7GKJIMwKXX58cbJUodDfI76qqUUfOxZYFl3TWn46m08)

*Figure 6.1 - Processor profiling*

Test setup:
* RTX3060 Laptop GPU
* AMD Ryzen 7 5800H CPU
* Fast NVMe SSD

Test data - Envisat IM -> Envisat IMS. About 16 seconds of signal data. The whole chain is processed in less than 2 seconds.
Summary of the time spent:
1. File read, metadata parsing & signal data DISK -> CPU -> GPU (350 ms)
2. Core SAR DSP chain (700 ms)
3. Metadata assembly, file writing GPU -> CPU -> disk (750 ms)

Steps 1 & 3 are heavily dependent on the performance of the storage and can be a major bottleneck. Step 2 is mostly done
on the GPU and the most time-consuming steps are the FFTs calculated by cuFFT. The CPU speed is not relevant as no
major focussing steps are bottleneck by the CPU.

There still some optimizations that could be done:
* Forecast level 0 dataset offsets and therefore no need to wait and read all the file
* Deallocate GPU and RAM buffers earlier

With this there could be nearly 1 second processing time when the RAM disk is a source and target disk storage.

## 6.2 GPU and RAM memory limit based on the dataset size

The baseline GPU memory requirement is estimated to be:\
`Total memory usage = Range size * Azimuth size * 8 * 1.3 * 2`

Where:
* Range size is level 0 range size
* Azimuth size is total number of processed packets
* 8 is size of the complex float
* 1.3 is rough estimate of the paddings required for range and azimuth compression and additional metadata
* 2 is memory used for FFTs and output image generation (that includes back and forth buffer)

On a standard 16 seconds IMS dataset, this results in about 3GB of GPU memory. CPU RAM memory usage is roughly
the input file size plus the output file size. Other elements take insignificant amount of memory.
