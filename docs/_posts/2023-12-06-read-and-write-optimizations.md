---
title: Read and write optimizations
date: 2023-12-06 10:00:00 +0200
categories: [optimizations, profiling]
tags: [nsys,metadata,samples,memory,io]
---



# Optimizations that enable efficient use of GPU

Case study based on SAR/ASAR instruments data focussing of the ERS and ENVISAT L0 datasets processing.\
Basic principles for efficient GPU computing in context of EO data processing:
* Make I/O efficient
  * Read/write as little as possible
  * Read/write async when possible
* Minimize data transfer between host/device (e.g. from RAM to GPU memory and vice versa)
* Implement as much as possible in GPU kernels, even if it might be counterintuitive in a sense of parallel processing and not connected to pure mathematical calculations e.g. parsing and converting

Below will be presented optimizations following those principles. No CPU resource/threads were used to speedup
calculations only lightweight threads to initialize something. At any given single point maximum of 1 background thread
was running. Maximum usage of GPU memory 2.8 GB (For 16 seconds of Envisat IM L0) remained the same. Envisat and ERS results
are similar, although the parsing structures are different.

The benchmark device used throughout the document (unless noted otherwise) has the following specification:
```
Lenovo Legion 5
CPU: Intel i7 10750h
RAM: 32GB
NVIDIA GeForce GTX 1660 Ti 6GB VRAM
Transcend MTE400S 1 TB M.2 NVMe SSD
```

Command and dataset used for the measurements
```bash
asar_focus --input ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1 --aux ASAR_Auxiliary_Files/ --orb DOR_VOR_AX_d/ -o /home/foo/ds_garden/asar_focus_envisat_ims -t IMS --log debug
```

**The story from ~3.5 seconds to ~1.3 seconds**

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

### Summary of measurements

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

**Before**
![nsys profile_before_optimizations](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/asar-focus-before-optimizations-bbaa0d3.png)
[Get the NSYS profiling report](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/report_asar_focus_before_optimizations_bbaa0d3_w_memory.nsys-rep)

**After**
![nsys_profile_after_write_optimizations](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/asar-focus-write-optimizations-599391f.png)
[Get the NSYS profile report](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/report_asar_focus_results_write_condition_599391f_w_memory.nsys-rep)

</details>

## Parse optimizations

Lastly greatest gain was achieved by parsing the L0 dataset. All the measurement data encapsulated in the L0 packets was
copied into buffer and from there straight to the GPU which stayed there until the start of the writing. For that some
custom approaches were created, for example prepending sample/measurement count to each range/row measurement sequence
for ASAR. Also forecasting maximum SWST before parsing by iterating through all the ASAR packets.

### FBAQ decompression

For ASAR decompression substep of converting measurements to complex float value was refactored. Following small function
was created at `alus::asar::envformat`
```c++
inline __device__ __host__ int Fbaq4IndexCalc(int block, int raw_index) {
    int converted_index;
    if (raw_index > 7) {
        converted_index = 15 - raw_index;
    } else {
        converted_index = 8 + raw_index;
    }

    return 256 * converted_index + block;
}
```

This would give the index to LUT table that is parsed from instrument's AUX file. Can be used on CPU and GPU.

### Samples to GPU buffer

This was achieved already in the parsing process. First all samples were gathered and then copied straight to GPU memory.
Next two kernels were created separately for ERS and ENVISAT in `alus::asar::envformat`

```c++
__global__ void ConvertErsImSamplesToComplexKernel(uint8_t* samples, int samples_range_size_bytes, int packets,
                                                   uint16_t* swst_codes, uint16_t min_swst, cufftComplex* target_buf,
                                                   int target_range_width) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int x_size = samples_range_size_bytes;

    if (x < x_size && y < packets) {
        const int buffer_index = y * x_size + x;
        const int target_buffer_index = y * target_range_width + (x / 2);
        // ERS SWST multiplier = 4
        int offset = 4 * (swst_codes[y] - min_swst);
        if (x % 2 == 0) {
            target_buf[target_buffer_index + offset].x = static_cast<float>(samples[buffer_index]);
        } else {
            target_buf[target_buffer_index + offset].y = static_cast<float>(samples[buffer_index]);
        }
    }
}

__global__ void ConvertAsarImBlocksToComplexKernel(uint8_t* block_samples, int block_samples_item_length_bytes,
                                                   int records, uint16_t* swst_codes, uint16_t min_swst,
                                                   cufftComplex* gpu_buffer, int target_range_padded_samples,
                                                   float* i_lut_fbaq4, float* q_lut_fbaq4, int lut_items) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int x_size = block_samples_item_length_bytes;

    if (x < x_size && y < records) {
        // These are the bytes that specify the length of raw data blocks item.
        if (x < 2) {
            return;
        }

        const auto block_samples_item_length_address = block_samples + y * block_samples_item_length_bytes;
        uint16_t block_samples_item_length{0};
        block_samples_item_length |= 0x00FF & block_samples_item_length_address[0];
        block_samples_item_length |= 0xFF00 & (static_cast<uint16_t>(block_samples_item_length_address[1]) << 8);

        const int block_set_sample_index = x - 2;
        // No data for this entry.
        if (block_set_sample_index >= block_samples_item_length) {
            return;
        }

        constexpr int BLOCK_SIZE{64};
        // This is the block ID
        if (block_set_sample_index % BLOCK_SIZE == 0) {
            return;
        }

        const uint8_t* blocks_start_address = block_samples_item_length_address + 2;
        const auto block_index = block_set_sample_index / BLOCK_SIZE;
        const uint8_t block_id = blocks_start_address[block_index * BLOCK_SIZE];
        const uint8_t codeword = blocks_start_address[block_set_sample_index];
        const uint8_t codeword_i = codeword >> 4;
        const uint8_t codeword_q = codeword & 0x0F;
        const auto i_lut_index = Fbaq4IndexCalc(block_id, codeword_i);
        const auto q_lut_index = Fbaq4IndexCalc(block_id, codeword_q);
        // This is an error somewhere.
        if (i_lut_index >= lut_items || q_lut_index >= lut_items) {
            return;
        }
        const auto i_sample = i_lut_fbaq4[i_lut_index];
        const auto q_sample = q_lut_fbaq4[q_lut_index];

        auto range_sample_index = block_set_sample_index - (block_index + 1);
        range_sample_index += (swst_codes[y] - min_swst);
        range_sample_index += (y * target_range_padded_samples);
        gpu_buffer[range_sample_index].x = i_sample;
        gpu_buffer[range_sample_index].y = q_sample;
    }
}
```

This required changing the existing parsing logic like accumulating SWST codes over the range/rows and of course copying
measurements straight to buffer without any decompression, unpacking or conversion.

Before the whole parsing took up to 1 second. Now it is less than 100 milliseconds. Converting kernels takes ~12 milliseconds.

### Summary of measurements

<details>
  <summary>Execution log before/after</summary>

**Before ~2.8 sec (same as above's writing optimization's "After")**
```bash
See above...
```

**After (~1.3 sec)**
```bash
[2023-12-05 16:43:55.016291] [0x00007fefa87f8000] [info]    asar_focus/0.1.2
[2023-12-05 16:43:55.091998] [0x00007fefa87f8000] [debug]   Input DS read and fetch time = 75 ms
[2023-12-05 16:43:55.110696] [0x00007fefa87f8000] [debug]   Orbit file fetch time = 17 ms
[2023-12-05 16:43:55.116191] [0x00007fefa87f8000] [info]    Instrument file = /home/foo/ds_garden/asar_focus_envisat_im_store/ASAR_Auxiliary_Files/ASA_INS_AX/ASA_INS_AXVIEC20061220_105425_20030211_000000_20071231_000000
[2023-12-05 16:43:55.177393] [0x00007fefa87f8000] [debug]   Product name = ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1
[2023-12-05 16:43:55.177425] [0x00007fefa87f8000] [debug]   Input SENSING_START=2004-Jan-09 19:49:24.000509
[2023-12-05 16:43:55.177434] [0x00007fefa87f8000] [debug]   Input SENSING_END=2004-Jan-09 19:49:42.000599
[2023-12-05 16:43:55.177441] [0x00007fefa87f8000] [debug]   ORBIT vectors start 2004-Jan-09 19:44:28
[2023-12-05 16:43:55.177446] [0x00007fefa87f8000] [debug]   ORBIT vectors end 2004-Jan-09 19:54:28
[2023-12-05 16:43:55.179813] [0x00007fefa87f8000] [debug]   Forecasted maximum sample blocks bytes/length per range 5772
[2023-12-05 16:43:55.179828] [0x00007fefa87f8000] [debug]   Reserving 163MiB for raw sample blocks buffer including prepending 2 byte length marker (5774x29744)
[2023-12-05 16:43:55.232713] [0x00007fefa87f8000] [debug]   Swath = IS2 idx = 1
[2023-12-05 16:43:55.232738] [0x00007fefa87f8000] [debug]   Chirp Kr = 588741148672, n_samp = 522
[2023-12-05 16:43:55.232742] [0x00007fefa87f8000] [debug]   tx pulse calc dur = 2.7176629348260696e-05
[2023-12-05 16:43:55.232744] [0x00007fefa87f8000] [debug]   Nominal chirp duration = 2.7176629373570904e-05
[2023-12-05 16:43:55.232747] [0x00007fefa87f8000] [debug]   Calculated chirp BW = 15999999.99442935 , meta bw = 16000000
[2023-12-05 16:43:55.232749] [0x00007fefa87f8000] [debug]   range samples = 5705, minimum range padded size = 6227
[2023-12-05 16:43:55.232751] [0x00007fefa87f8000] [debug]   n PRI before SWST = 9
[2023-12-05 16:43:55.232754] [0x00007fefa87f8000] [debug]   SWST changes 1 MIN|MAX SWST 2024|2048
[2023-12-05 16:43:55.232772] [0x00007fefa87f8000] [debug]   PRF 1652.42 PRI 0.000605175
[2023-12-05 16:43:55.232776] [0x00007fefa87f8000] [debug]   Carrier frequency: 5331004416 Range sampling rate = 19207680
[2023-12-05 16:43:55.232778] [0x00007fefa87f8000] [debug]   Slant range to first sample = 832215.7522699253 m two way slant time 5551945.888311342 ns
[2023-12-05 16:43:55.232782] [0x00007fefa87f8000] [debug]   platform velocity = 7545.17, initial Vr = 7078
[2023-12-05 16:43:55.432756] [0x00007fefa87f8000] [debug]   nadirs start 57.025928 19.417803 - stop  63.840936 14.52136
[2023-12-05 16:43:55.432766] [0x00007fefa87f8000] [debug]   guess = 60.433431999999996 22.417803
[2023-12-05 16:43:55.432790] [0x00007fefa87f8000] [debug]   center point = 58.8427 23.603
[2023-12-05 16:43:55.432793] [0x00007fefa87f8000] [debug]   Diagnostic summary of the packets
[2023-12-05 16:43:55.432795] [0x00007fefa87f8000] [debug]   Calibration packets - 29
[2023-12-05 16:43:55.432797] [0x00007fefa87f8000] [debug]   Seq ctrl [oos total] 0 0
[2023-12-05 16:43:55.432799] [0x00007fefa87f8000] [debug]   Mode packet [oos total] 0 0
[2023-12-05 16:43:55.432801] [0x00007fefa87f8000] [debug]   Cycle packet [oos total] 0 0
[2023-12-05 16:43:55.432803] [0x00007fefa87f8000] [debug]   MIN|MAX datablock lengths in bytes 5772|5772
[2023-12-05 16:43:55.432872] [0x00007fefa87f8000] [debug]   LVL0 file packet parse time = 322 ms
[2023-12-05 16:43:55.432979] [0x00007fefa87f8000] [debug]   Range FFT padding sz = 6250(2 ^ 1 * 3 ^ 0 * 5 ^ 5 * 7 ^ 0)
[2023-12-05 16:43:55.432982] [0x00007fefa87f8000] [debug]   azimuth FFT padding sz = 30000 (2 ^ 4 * 3 ^ 1 * 5 ^ 4 * 7 ^ 0)
[2023-12-05 16:43:55.432985] [0x00007fefa87f8000] [debug]   FFT paddings: rg  = 5705->6250 az = 29743->30000
[2023-12-05 16:43:55.452475] [0x00007fefa87f8000] [debug]   GPU image formation time = 19 ms
[2023-12-05 16:43:55.470326] [0x00007fefa87f8000] [debug]   DC bias = 0.000386579 0.000622201
[2023-12-05 16:43:55.526863] [0x00007fefa87f8000] [debug]   quad misconf = 2.14816
[2023-12-05 16:43:55.539898] [0x00007fefa87f8000] [debug]   I/Q correction time = 87 ms
[2023-12-05 16:43:55.539983] [0x00007fefa87f8000] [debug]   Vr estimation time = 0 ms
[2023-12-05 16:43:55.564514] [0x00007fefa87f8000] [debug]   fractional DC estimation time = 24 ms
[2023-12-05 16:43:55.564526] [0x00007fefa87f8000] [debug]   Image GPU byte size = 1.5GB
[2023-12-05 16:43:55.564529] [0x00007fefa87f8000] [debug]   Estimated GPU memory usage ~3GB
[2023-12-05 16:43:55.680710] [0x00007fefa87f8000] [debug]   Range compression time = 114 ms
[2023-12-05 16:43:55.935424] [0x00007fefa87f8000] [debug]   Azimuth compression time = 254 ms
[2023-12-05 16:43:55.944704] [0x00007fefa87f8000] [debug]   Image results correction time = 9 ms
[2023-12-05 16:43:56.021283] [0x00007fefa87f8000] [debug]   MDS Host buffer transfer time = 76 ms
[2023-12-05 16:43:56.350812] [0x00007fefa87f8000] [debug]   LVL1 file write time = 329 ms
```

</details>

<details>
  <summary>Profiling charts before/after</summary>

**Before (same as above's writing optimization's "After")**
![nsys_profile_after_write_optimizations](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/asar-focus-write-optimizations-599391f.png)
[Get the NSYS profile report](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/report_asar_focus_results_write_condition_599391f_w_memory.nsys-rep)

**After**
![nsys_profile_after_parse_optimizations](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/asar-focus-parse-prepare-optimizations-1bd6dc16.png)
[Get the NSYS profile report](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/report_asar_focus_parse_prepare_optimizations_1bd6dc16_w_memory.nsys-rep)

</details>

## Caveats

All tests were performed with cached/swapped input datasets (up to 100 milliseconds to read L0 into buffer).
The results could take more time, for example as can be seen below:

```bash
[2023-12-06 18:14:33.538486] [0x00007f00ecb85000] [info]    asar_focus/0.1.2
[2023-12-06 18:14:34.408000] [0x00007f00ecb85000] [debug]   Input DS read and fetch time = 869 ms
[2023-12-06 18:14:34.528318] [0x00007f00ecb85000] [debug]   Orbit file fetch time = 46 ms
[2023-12-06 18:14:34.537565] [0x00007f00ecb85000] [info]    Instrument file = /home/foo/ds_garden/asar_focus_envisat_im_store/ASAR_Auxiliary_Files/ASA_INS_AX/ASA_INS_AXVIEC20061220_105425_20030211_000000_20071231_000000
[2023-12-06 18:14:34.538483] [0x00007f00ecb85000] [debug]   Product name = ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1
[2023-12-06 18:14:34.538506] [0x00007f00ecb85000] [debug]   Input SENSING_START=2004-Jan-09 19:49:24.000509
[2023-12-06 18:14:34.538516] [0x00007f00ecb85000] [debug]   Input SENSING_END=2004-Jan-09 19:49:42.000599
[2023-12-06 18:14:34.538523] [0x00007f00ecb85000] [debug]   ORBIT vectors start 2004-Jan-09 19:44:28
[2023-12-06 18:14:34.538529] [0x00007f00ecb85000] [debug]   ORBIT vectors end 2004-Jan-09 19:54:28
[2023-12-06 18:14:34.540888] [0x00007f00ecb85000] [debug]   Forecasted maximum sample blocks bytes/length per range 5772
[2023-12-06 18:14:34.540898] [0x00007f00ecb85000] [debug]   Reserving 163MiB for raw sample blocks buffer including prepending 2 byte length marker (5774x29744)
[2023-12-06 18:14:34.596635] [0x00007f00ecb85000] [debug]   Swath = IS2 idx = 1
[2023-12-06 18:14:34.596660] [0x00007f00ecb85000] [debug]   Chirp Kr = 588741148672, n_samp = 522
[2023-12-06 18:14:34.596664] [0x00007f00ecb85000] [debug]   tx pulse calc dur = 2.7176629348260696e-05
[2023-12-06 18:14:34.596667] [0x00007f00ecb85000] [debug]   Nominal chirp duration = 2.7176629373570904e-05
[2023-12-06 18:14:34.596671] [0x00007f00ecb85000] [debug]   Calculated chirp BW = 15999999.99442935 , meta bw = 16000000
[2023-12-06 18:14:34.596674] [0x00007f00ecb85000] [debug]   range samples = 5705, minimum range padded size = 6227
[2023-12-06 18:14:34.596677] [0x00007f00ecb85000] [debug]   n PRI before SWST = 9
[2023-12-06 18:14:34.596680] [0x00007f00ecb85000] [debug]   SWST changes 1 MIN|MAX SWST 2024|2048
[2023-12-06 18:14:34.598370] [0x00007f00ecb85000] [debug]   PRF 1652.42 PRI 0.000605175
[2023-12-06 18:14:34.598378] [0x00007f00ecb85000] [debug]   Carrier frequency: 5331004416 Range sampling rate = 19207680
[2023-12-06 18:14:34.598381] [0x00007f00ecb85000] [debug]   Slant range to first sample = 832215.7522699253 m two way slant time 5551945.888311342 ns
[2023-12-06 18:14:34.598385] [0x00007f00ecb85000] [debug]   platform velocity = 7545.17, initial Vr = 7078
[2023-12-06 18:14:34.776462] [0x00007f00ecb85000] [debug]   nadirs start 57.025928 19.417803 - stop  63.840936 14.52136
[2023-12-06 18:14:34.776476] [0x00007f00ecb85000] [debug]   guess = 60.433431999999996 22.417803
[2023-12-06 18:14:34.776513] [0x00007f00ecb85000] [debug]   center point = 58.8427 23.603
[2023-12-06 18:14:34.776516] [0x00007f00ecb85000] [debug]   Diagnostic summary of the packets
[2023-12-06 18:14:34.776520] [0x00007f00ecb85000] [debug]   Calibration packets - 29
[2023-12-06 18:14:34.776523] [0x00007f00ecb85000] [debug]   Seq ctrl [oos total] 0 0
[2023-12-06 18:14:34.776525] [0x00007f00ecb85000] [debug]   Mode packet [oos total] 0 0
[2023-12-06 18:14:34.776529] [0x00007f00ecb85000] [debug]   Cycle packet [oos total] 0 0
[2023-12-06 18:14:34.776533] [0x00007f00ecb85000] [debug]   MIN|MAX datablock lengths in bytes 5772|5772
[2023-12-06 18:14:34.776541] [0x00007f00ecb85000] [debug]   LVL0 file packet parse time = 248 ms
[2023-12-06 18:14:34.776622] [0x00007f00ecb85000] [debug]   Range FFT padding sz = 6250(2 ^ 1 * 3 ^ 0 * 5 ^ 5 * 7 ^ 0)
[2023-12-06 18:14:34.776626] [0x00007f00ecb85000] [debug]   azimuth FFT padding sz = 30000 (2 ^ 4 * 3 ^ 1 * 5 ^ 4 * 7 ^ 0)
[2023-12-06 18:14:34.776629] [0x00007f00ecb85000] [debug]   FFT paddings: rg  = 5705->6250 az = 29743->30000
[2023-12-06 18:14:34.795463] [0x00007f00ecb85000] [debug]   GPU image formation time = 18 ms
[2023-12-06 18:14:34.809561] [0x00007f00ecb85000] [debug]   DC bias = 0.000386579 0.000622201
[2023-12-06 18:14:34.867000] [0x00007f00ecb85000] [debug]   quad misconf = 2.14816
[2023-12-06 18:14:34.880021] [0x00007f00ecb85000] [debug]   I/Q correction time = 84 ms
[2023-12-06 18:14:34.880115] [0x00007f00ecb85000] [debug]   Vr estimation time = 0 ms
[2023-12-06 18:14:34.902472] [0x00007f00ecb85000] [debug]   fractional DC estimation time = 22 ms
[2023-12-06 18:14:34.902487] [0x00007f00ecb85000] [debug]   Image GPU byte size = 1.5GB
[2023-12-06 18:14:34.902491] [0x00007f00ecb85000] [debug]   Estimated GPU memory usage ~3GB
[2023-12-06 18:14:35.034953] [0x00007f00ecb85000] [debug]   Range compression time = 130 ms
[2023-12-06 18:14:35.287949] [0x00007f00ecb85000] [debug]   Azimuth compression time = 252 ms
[2023-12-06 18:14:35.296819] [0x00007f00ecb85000] [debug]   Image results correction time = 8 ms
[2023-12-06 18:14:35.398208] [0x00007f00ecb85000] [debug]   MDS Host buffer transfer time = 101 ms
[2023-12-06 18:14:35.820696] [0x00007f00ecb85000] [debug]   LVL1 file write time = 422 ms
```

A whopping 2.8 seconds! This is all influenced by reading in the L0 dataset (second line in the log - 869 ms). Future optimizations will try to enhance this
especially for L0 datasets that have long sensing period - no need to read all the packets since usually published
IMS L1 data is 18 seconds long. For production pipelines it is advised to use ramdisk to store the input L0 datasets and
also L1 ones, which would speed up writing (results presented here could reach 1 second if such strategy would have been used).

Below is an example of the [AWS G3 instance](https://aws.amazon.com/ec2/instance-types/g3/) for the same dataset and
input arguments. It has slower CPU and older generation Tesla M60.

```bash
[2023-12-04 13:30:17.466873] [0x00007ff136537000] [info]    asar_focus/0.1.2
[2023-12-04 13:30:17.645265] [0x00007ff136537000] [debug]   Input DS read and fetch time = 178 ms
[2023-12-04 13:30:17.755409] [0x00007ff136537000] [debug]   Orbit file fetch time = 51 ms
[2023-12-04 13:30:17.765069] [0x00007ff136537000] [info]    Instrument file = /root/e2e/envisat/ASAR_Auxiliary_Files/ASA_INS_AX/ASA_INS_AXVIEC20061220_105425_20030211_000000_20071231_000000
[2023-12-04 13:30:17.765590] [0x00007ff136537000] [debug]   Product name = ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1
[2023-12-04 13:30:17.765642] [0x00007ff136537000] [debug]   Input SENSING_START=2004-Jan-09 19:49:24.000509
[2023-12-04 13:30:17.765671] [0x00007ff136537000] [debug]   Input SENSING_END=2004-Jan-09 19:49:42.000599
[2023-12-04 13:30:17.765686] [0x00007ff136537000] [debug]   ORBIT vectors start 2004-Jan-09 19:44:28
[2023-12-04 13:30:17.765705] [0x00007ff136537000] [debug]   ORBIT vectors end 2004-Jan-09 19:54:28
[2023-12-04 13:30:17.769317] [0x00007ff136537000] [debug]   Forecasted maximum sample blocks bytes/length per range 5772
[2023-12-04 13:30:17.769341] [0x00007ff136537000] [debug]   Reserving 163MiB for raw sample blocks buffer including prepending 2 byte length marker (5774x29744)
[2023-12-04 13:30:17.905393] [0x00007ff136537000] [debug]   Swath = IS2 idx = 1
[2023-12-04 13:30:17.905449] [0x00007ff136537000] [debug]   Chirp Kr = 588741148672, n_samp = 522
[2023-12-04 13:30:17.905459] [0x00007ff136537000] [debug]   tx pulse calc dur = 2.7176629348260696e-05
[2023-12-04 13:30:17.905464] [0x00007ff136537000] [debug]   Nominal chirp duration = 2.7176629373570904e-05
[2023-12-04 13:30:17.905471] [0x00007ff136537000] [debug]   Calculated chirp BW = 15999999.99442935 , meta bw = 16000000
[2023-12-04 13:30:17.905476] [0x00007ff136537000] [debug]   range samples = 5705, minimum range padded size = 6227
[2023-12-04 13:30:17.905482] [0x00007ff136537000] [debug]   n PRI before SWST = 9
[2023-12-04 13:30:17.905495] [0x00007ff136537000] [debug]   SWST changes 1 MIN|MAX SWST 2024|2048
[2023-12-04 13:30:17.905511] [0x00007ff136537000] [debug]   PRF 1652.42 PRI 0.000605175
[2023-12-04 13:30:17.905523] [0x00007ff136537000] [debug]   Carrier frequency: 5331004416 Range sampling rate = 19207680
[2023-12-04 13:30:17.905529] [0x00007ff136537000] [debug]   Slant range to first sample = 832215.7522699253 m two way slant time 5551945.888311342 ns
[2023-12-04 13:30:17.905550] [0x00007ff136537000] [debug]   platform velocity = 7545.17, initial Vr = 7078
[2023-12-04 13:30:18.280392] [0x00007ff136537000] [debug]   nadirs start 57.025928 19.417803 - stop  63.840936 14.52136
[2023-12-04 13:30:18.280433] [0x00007ff136537000] [debug]   guess = 60.433431999999996 22.417803
[2023-12-04 13:30:18.281466] [0x00007ff136537000] [debug]   center point = 58.8427 23.603
[2023-12-04 13:30:18.281486] [0x00007ff136537000] [debug]   Diagnostic summary of the packets
[2023-12-04 13:30:18.281492] [0x00007ff136537000] [debug]   Calibration packets - 29
[2023-12-04 13:30:18.281504] [0x00007ff136537000] [debug]   Seq ctrl [oos total] 0 0
[2023-12-04 13:30:18.281515] [0x00007ff136537000] [debug]   Mode packet [oos total] 0 0
[2023-12-04 13:30:18.281521] [0x00007ff136537000] [debug]   Cycle packet [oos total] 0 0
[2023-12-04 13:30:18.281532] [0x00007ff136537000] [debug]   MIN|MAX datablock lengths in bytes 5772|5772
[2023-12-04 13:30:18.281741] [0x00007ff136537000] [debug]   LVL0 file packet parse time = 526 ms
[2023-12-04 13:30:18.281903] [0x00007ff136537000] [debug]   Range FFT padding sz = 6250(2 ^ 1 * 3 ^ 0 * 5 ^ 5 * 7 ^ 0)
[2023-12-04 13:30:18.281920] [0x00007ff136537000] [debug]   azimuth FFT padding sz = 30000 (2 ^ 4 * 3 ^ 1 * 5 ^ 4 * 7 ^ 0)
[2023-12-04 13:30:18.281926] [0x00007ff136537000] [debug]   FFT paddings: rg  = 5705->6250 az = 29743->30000
[2023-12-04 13:30:18.328962] [0x00007ff136537000] [debug]   GPU image formation time = 47 ms
[2023-12-04 13:30:18.350327] [0x00007ff136537000] [debug]   DC bias = 0.000386579 0.000622201
[2023-12-04 13:30:18.451795] [0x00007ff136537000] [debug]   quad misconf = 2.14816
[2023-12-04 13:30:18.480800] [0x00007ff136537000] [debug]   I/Q correction time = 151 ms
[2023-12-04 13:30:18.481014] [0x00007ff136537000] [debug]   Vr estimation time = 0 ms
[2023-12-04 13:30:18.537408] [0x00007ff136537000] [debug]   fractional DC estimation time = 56 ms
[2023-12-04 13:30:18.537422] [0x00007ff136537000] [debug]   Image GPU byte size = 1.5GB
[2023-12-04 13:30:18.537435] [0x00007ff136537000] [debug]   Estimated GPU memory usage ~3GB
[2023-12-04 13:30:18.894944] [0x00007ff136537000] [debug]   Range compression time = 354 ms
[2023-12-04 13:30:19.424892] [0x00007ff136537000] [debug]   Azimuth compression time = 529 ms
[2023-12-04 13:30:19.442874] [0x00007ff136537000] [debug]   Image results correction time = 17 ms
[2023-12-04 13:30:19.501334] [0x00007ff136537000] [debug]   MDS Host buffer transfer time = 58 ms
[2023-12-04 13:30:19.805665] [0x00007ff136537000] [debug]   LVL1 file write time = 304 ms
```

This took ~2.4 seconds. Mainly because of the slower GPU (for example azimuth compression 529 vs 252 milliseconds) and parsing
(which took 300 milliseconds, 200 more than in the examples before)