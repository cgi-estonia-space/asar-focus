---
title: Optimizations
date: 2023-12-06 10:00:00 +0200
categories: [optimizations, profiling]
tags: [nsys,metadata,samples,memory,io]
---



# Optimizations that enable efficient use of GPU

Basic principles for efficient GPU computing in context of EO data processing:
* Make I/O efficient
  * Read/write as little as possible
  * Read/write async when possible
* Minimize data transfer between host/device (e.g. from RAM to GPU memory and vice versa)
* Implement as much as possible in GPU kernels (even if it might be counterintuitive and not connected to pure mathematical calculations)

Below will be presented optimizations following those principles.

## Write/output optimizations

Tere
