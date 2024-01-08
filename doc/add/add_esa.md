


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

The Envisat format support handling and support was implemented during this project. No libraries were used to handle
this.

asar_focus processor:
* fmt -> formating of messages
* boost_log -> system logs
* boost_program_arguments -> CLI
* GPU memory management and transfer -> **CUDA**


# 5 Component Description

## 5.1 DevicePaddedImage

SAR focusing requires forward and backwards Fourier transforms in range and azimuth directions, additionally the SAR datasets are quite large

DevicePaddedImage is the core abstraction to facilitate efficient SAR transformations by embedding extra padding areas in the right and bottom of the image. This allows inplace FFTs in both range and azimuth direction, this allievates the need for extra memory copies to fullfill FFT size requirements. Thus changing from time domain to frequency domain and back is done memory efficiently, without needlessly copying multigigabyte signal data.

Additional design consideration is the fact that GPUs have less memory than CPUs, thus using this techinique allows to keep the full DSP chain on the GPU memory, avoiding costly CPU <-> GPU transfers.
Visual representation of the memory layout.
<pre>
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
</pre>

# 6 Feasability and Resource Estimates

Profiling analyze

GPU and RAM memory limit based on the dataset size



