---
title: Read and write optimizations
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
* Implement as much as possible in GPU kernels, even if it might be counterintuitive in a sense of parallel processing and not connected to pure mathematical calculations e.g. parsing and converting

Below will be presented optimizations following those principles.

The benchmark device used throughout the document (unless noted otherwise) has the following specification:
```
Lenovo Legion 5
CPU: Intel i7 10750h
RAM: 32GB
NVIDIA GeForce GTX 1660 Ti 6GB VRAM
Transcend MTE400S 1 TB M.2 NVMe SSD
```

## Write/output optimizations

First optimizations implemented were approached from bottom to top. Below will be described what were done in order to
minimize latency and increase throughput after the compression pipeline has finished.

### Compressed I/Q pixels condition

* Calibrate with the determined constant
* Clamp value to [-32768, 32767]
* Convert from floats to complex INT16 values
* Format to Big-Endian as required by the Envisat format
* Package each range/row with a header as specified by L1 format

This is all done in a single kernel called 
```c++
CalibrateClampToBigEndian(cufftComplex* data, int x_size_stride, int x_size, int y_size,char* dest_space, int record_header_size, float calibration_constant)
```

This operation took around 400 ms on single thread CPU loop. Running it in a kernel takes around 10 ms. As a side note -
transferring same results to RAM in order to further writes takes around 80 ms. 

### MDS initialization

This is a buffer that stores the records (results of the previously described step) that are finally written to the
resulting L1 file. The count of the samples/range and azimuth size is determined during the parsing. When allocating
a buffer the real pages which are assigned/used by the OS are not actually initialized. Hence there was series of 
experiments to determine the fastest scenario that would provide the faster writing/store speeds. The experiments are
published in a separate [alus-experiments repo](https://github.com/cgi-estonia-space/alus-experiments). The async
memory allocation and page initialization is implemented in
```c++
namespace alus::util {
// Refer to "alus-experiments" host_memory_allocation.cc for speed tests.
inline void* Memalloc(size_t bytes) {
    ...
}
```

This is called after parsing and when moving results calculated on GPU to RAM takes around 80 ms instead of ~360ms as 
before.

### IMS async write

This is mostly metadata and dataset descriptions. Such an information could be written after azimuth compression.
Async writing is accomplished with the following class
```c++
alus::util::FileAsync
```

This is not large data, but around 150 ms was saved because this happens during the calculations and assembly.

### 


<details>
  <summary>Execution log before/after</summary>

**Before (~3.5 sec)**
```bash
[2023-12-05 16:24:49.309104] [0x00007f6ca3c42000] [info]    asar_focus/0.1.2
[2023-12-05 16:24:49.385988] [0x00007f6ca3c42000] [debug]   Input DS read and fetch time = 76 ms
[2023-12-05 16:24:49.406430] [0x00007f6ca3c42000] [debug]   Orbit file fetch time = 19 ms
[2023-12-05 16:24:49.411846] [0x00007f6ca3c42000] [info]    Instrument file = /home/foo/ds_garden/asar_focus_envisat_im_store/ASAR_Auxiliary_Files/ASA_INS_AX/ASA_INS_AXVIEC20061220_105425_20030211_000000_20071231_000000
[2023-12-05 16:24:49.469142] [0x00007f6ca3c42000] [debug]   Product name = ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1
[2023-12-05 16:24:49.469243] [0x00007f6ca3c42000] [debug]   Input SENSING_START=2004-Jan-09 19:49:24.000509
[2023-12-05 16:24:49.469272] [0x00007f6ca3c42000] [debug]   Input SENSING_END=2004-Jan-09 19:49:42.000599
[2023-12-05 16:24:49.469280] [0x00007f6ca3c42000] [debug]   ORBIT vectors start 2004-Jan-09 19:44:28
[2023-12-05 16:24:49.469286] [0x00007f6ca3c42000] [debug]   ORBIT vectors end 2004-Jan-09 19:54:28
[2023-12-05 16:24:50.250152] [0x00007f6ca3c42000] [debug]   Swath = IS2 idx = 1
[2023-12-05 16:24:50.250173] [0x00007f6ca3c42000] [debug]   Chirp Kr = 588741148672, n_samp = 522
[2023-12-05 16:24:50.250176] [0x00007f6ca3c42000] [debug]   tx pulse calc dur = 2.7176629348260696e-05
[2023-12-05 16:24:50.250178] [0x00007f6ca3c42000] [debug]   Nominal chirp duration = 2.7176629373570904e-05
[2023-12-05 16:24:50.250181] [0x00007f6ca3c42000] [debug]   Calculated chirp BW = 15999999.99442935 , meta bw = 16000000
[2023-12-05 16:24:50.250183] [0x00007f6ca3c42000] [debug]   range samples = 5706, minimum range padded size = 6228
[2023-12-05 16:24:50.250186] [0x00007f6ca3c42000] [debug]   n PRI before SWST = 9
[2023-12-05 16:24:50.250230] [0x00007f6ca3c42000] [debug]   PRF 1652.42 PRI 0.000605175
[2023-12-05 16:24:50.250233] [0x00007f6ca3c42000] [debug]   Carrier frequency: 5331004416 Range sampling rate = 19207680
[2023-12-05 16:24:50.250235] [0x00007f6ca3c42000] [debug]   Slant range to first sample = 832215.7522699253 m two way slant time 5551945.888311342 ns
[2023-12-05 16:24:50.250259] [0x00007f6ca3c42000] [debug]   platform velocity = 7545.17, initial Vr = 7078
[2023-12-05 16:24:50.727977] [0x00007f6ca3c42000] [debug]   nadirs start 57.025928 19.417803 - stop  63.840936 14.52136
[2023-12-05 16:24:50.727998] [0x00007f6ca3c42000] [debug]   guess = 60.433431999999996 22.417803
[2023-12-05 16:24:50.728053] [0x00007f6ca3c42000] [debug]   center point = 58.8427 23.6033
[2023-12-05 16:24:50.728056] [0x00007f6ca3c42000] [debug]   Diagnostic summary of the packets
[2023-12-05 16:24:50.728059] [0x00007f6ca3c42000] [debug]   Calibration packets - 29
[2023-12-05 16:24:50.728061] [0x00007f6ca3c42000] [debug]   Seq ctrl [oos total] 0 0
[2023-12-05 16:24:50.728063] [0x00007f6ca3c42000] [debug]   Mode packet [oos total] 0 0
[2023-12-05 16:24:50.728064] [0x00007f6ca3c42000] [debug]   Cycle packet [oos total] 0 0
[2023-12-05 16:24:50.778760] [0x00007f6ca3c42000] [debug]   LVL0 file packet parse time = 1372 ms
[2023-12-05 16:24:50.778912] [0x00007f6ca3c42000] [debug]   Range FFT padding sz = 6250(2 ^ 1 * 3 ^ 0 * 5 ^ 5 * 7 ^ 0)
[2023-12-05 16:24:50.778917] [0x00007f6ca3c42000] [debug]   azimuth FFT padding sz = 30000 (2 ^ 4 * 3 ^ 1 * 5 ^ 4 * 7 ^ 0)
[2023-12-05 16:24:50.778940] [0x00007f6ca3c42000] [debug]   FFT paddings: rg  = 5706->6250 az = 29743->30000
[2023-12-05 16:24:51.107974] [0x00007f6ca3c42000] [debug]   GPU image formation time = 329 ms
[2023-12-05 16:24:51.124038] [0x00007f6ca3c42000] [debug]   DC bias = 0.000388305 0.000623866
[2023-12-05 16:24:51.179383] [0x00007f6ca3c42000] [debug]   quad misconf = 2.14832
[2023-12-05 16:24:51.191305] [0x00007f6ca3c42000] [debug]   I/Q correction time = 83 ms
[2023-12-05 16:24:51.191386] [0x00007f6ca3c42000] [debug]   Vr estimation time = 0 ms
[2023-12-05 16:24:51.218092] [0x00007f6ca3c42000] [debug]   fractional DC estimation time = 26 ms
[2023-12-05 16:24:51.218126] [0x00007f6ca3c42000] [debug]   Image GPU byte size = 1.5GB
[2023-12-05 16:24:51.218129] [0x00007f6ca3c42000] [debug]   Estimated GPU memory usage ~3GB
[2023-12-05 16:24:51.335384] [0x00007f6ca3c42000] [debug]   Range compression time = 116 ms
[2023-12-05 16:24:51.589792] [0x00007f6ca3c42000] [debug]   Azimuth compression time = 254 ms
[2023-12-05 16:24:52.085189] [0x00007f6ca3c42000] [debug]   Image GPU->CPU time = 493 ms
[2023-12-05 16:24:52.448888] [0x00007f6ca3c42000] [debug]   MDS construction time = 363 ms
[2023-12-05 16:24:52.872295] [0x00007f6ca3c42000] [info]    Focussed product saved at /home/foo/ds_garden/asar_focus_envisat_ims/ASA_IMS_1PNPDK20040109_194924_000000182023_00157_09730_1479.N1
[2023-12-05 16:24:52.872452] [0x00007f6ca3c42000] [debug]   LVL1 file write time = 423 ms
```

**After (~2.8 sec)**
```bash
[2023-12-05 16:39:47.772646] [0x00007f2915e25000] [info]    asar_focus/0.1.2
[2023-12-05 16:39:47.849456] [0x00007f2915e25000] [debug]   Input DS read and fetch time = 76 ms
[2023-12-05 16:39:47.867858] [0x00007f2915e25000] [debug]   Orbit file fetch time = 17 ms
[2023-12-05 16:39:47.876246] [0x00007f2915e25000] [info]    Instrument file = /home/foo/ds_garden/asar_focus_envisat_im_store/ASAR_Auxiliary_Files/ASA_INS_AX/ASA_INS_AXVIEC20061220_105425_20030211_000000_20071231_000000
[2023-12-05 16:39:47.938435] [0x00007f2915e25000] [debug]   Product name = ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1
[2023-12-05 16:39:47.938544] [0x00007f2915e25000] [debug]   Input SENSING_START=2004-Jan-09 19:49:24.000509
[2023-12-05 16:39:47.938582] [0x00007f2915e25000] [debug]   Input SENSING_END=2004-Jan-09 19:49:42.000599
[2023-12-05 16:39:47.938593] [0x00007f2915e25000] [debug]   ORBIT vectors start 2004-Jan-09 19:44:28
[2023-12-05 16:39:47.938602] [0x00007f2915e25000] [debug]   ORBIT vectors end 2004-Jan-09 19:54:28
[2023-12-05 16:39:48.729693] [0x00007f2915e25000] [debug]   Swath = IS2 idx = 1
[2023-12-05 16:39:48.729720] [0x00007f2915e25000] [debug]   Chirp Kr = 588741148672, n_samp = 522
[2023-12-05 16:39:48.729724] [0x00007f2915e25000] [debug]   tx pulse calc dur = 2.7176629348260696e-05
[2023-12-05 16:39:48.729726] [0x00007f2915e25000] [debug]   Nominal chirp duration = 2.7176629373570904e-05
[2023-12-05 16:39:48.729729] [0x00007f2915e25000] [debug]   Calculated chirp BW = 15999999.99442935 , meta bw = 16000000
[2023-12-05 16:39:48.729731] [0x00007f2915e25000] [debug]   range samples = 5706, minimum range padded size = 6228
[2023-12-05 16:39:48.729762] [0x00007f2915e25000] [debug]   n PRI before SWST = 9
[2023-12-05 16:39:48.729777] [0x00007f2915e25000] [debug]   PRF 1652.42 PRI 0.000605175
[2023-12-05 16:39:48.729781] [0x00007f2915e25000] [debug]   Carrier frequency: 5331004416 Range sampling rate = 19207680
[2023-12-05 16:39:48.729803] [0x00007f2915e25000] [debug]   Slant range to first sample = 832215.7522699253 m two way slant time 5551945.888311342 ns
[2023-12-05 16:39:48.729806] [0x00007f2915e25000] [debug]   platform velocity = 7545.17, initial Vr = 7078
[2023-12-05 16:39:49.221330] [0x00007f2915e25000] [debug]   nadirs start 57.025928 19.417803 - stop  63.840936 14.52136
[2023-12-05 16:39:49.221358] [0x00007f2915e25000] [debug]   guess = 60.433431999999996 22.417803
[2023-12-05 16:39:49.221420] [0x00007f2915e25000] [debug]   center point = 58.8427 23.6033
[2023-12-05 16:39:49.221423] [0x00007f2915e25000] [debug]   Diagnostic summary of the packets
[2023-12-05 16:39:49.221426] [0x00007f2915e25000] [debug]   Calibration packets - 29
[2023-12-05 16:39:49.221428] [0x00007f2915e25000] [debug]   Seq ctrl [oos total] 0 0
[2023-12-05 16:39:49.221430] [0x00007f2915e25000] [debug]   Mode packet [oos total] 0 0
[2023-12-05 16:39:49.221432] [0x00007f2915e25000] [debug]   Cycle packet [oos total] 0 0
[2023-12-05 16:39:49.269535] [0x00007f2915e25000] [debug]   LVL0 file packet parse time = 1401 ms
[2023-12-05 16:39:49.269698] [0x00007f2915e25000] [debug]   Range FFT padding sz = 6250(2 ^ 1 * 3 ^ 0 * 5 ^ 5 * 7 ^ 0)
[2023-12-05 16:39:49.269702] [0x00007f2915e25000] [debug]   azimuth FFT padding sz = 30000 (2 ^ 4 * 3 ^ 1 * 5 ^ 4 * 7 ^ 0)
[2023-12-05 16:39:49.269724] [0x00007f2915e25000] [debug]   FFT paddings: rg  = 5706->6250 az = 29743->30000
[2023-12-05 16:39:49.600809] [0x00007f2915e25000] [debug]   GPU image formation time = 331 ms
[2023-12-05 16:39:49.615898] [0x00007f2915e25000] [debug]   DC bias = 0.000388305 0.000623866
[2023-12-05 16:39:49.669358] [0x00007f2915e25000] [debug]   quad misconf = 2.14832
[2023-12-05 16:39:49.682269] [0x00007f2915e25000] [debug]   I/Q correction time = 81 ms
[2023-12-05 16:39:49.682374] [0x00007f2915e25000] [debug]   Vr estimation time = 0 ms
[2023-12-05 16:39:49.709786] [0x00007f2915e25000] [debug]   fractional DC estimation time = 27 ms
[2023-12-05 16:39:49.709798] [0x00007f2915e25000] [debug]   Image GPU byte size = 1.5GB
[2023-12-05 16:39:49.709801] [0x00007f2915e25000] [debug]   Estimated GPU memory usage ~3GB
[2023-12-05 16:39:49.825895] [0x00007f2915e25000] [debug]   Range compression time = 115 ms
[2023-12-05 16:39:50.080374] [0x00007f2915e25000] [debug]   Azimuth compression time = 254 ms
[2023-12-05 16:39:50.089725] [0x00007f2915e25000] [debug]   Image results correction time = 9 ms
[2023-12-05 16:39:50.170283] [0x00007f2915e25000] [debug]   MDS Host buffer transfer time = 80 ms
[2023-12-05 16:39:50.519320] [0x00007f2915e25000] [debug]   LVL1 file write time = 349 ms
```
</details>

<details>
  <summary>Profiling charts before/after</summary>


</details>
