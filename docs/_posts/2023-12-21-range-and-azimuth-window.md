---
title: Range and azimuth compression windowing
date: 2023-12-21 10:00:00 +0200
categories: [compression, focussing]
tags: [range,azimuth,metadata,samples]
---

# Background

Besides precise range and azimuth compression techniques, which is out of the focus here, it is necessary to
compensate range and azimuth artifacts that occur at the edges of the focussed results. 

When compressing all the samples/lines of a RAW or L0 dataset there would be echo artifacts/mirrors at the
boundaries of the azimuth. Either on top, bottom or both sides of the image. This is influenced by the doppler centroid
and azimuth modulation rate values.

An example below can be seen where echo replicas are seen at the bottom.
These were the initial results published by versions 0.1.X. The input dataset is **ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1**
![echo replicas](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040109_194924_000000182023_00157_09730_1479_echo_replicas.png)

Same example when terrain corrected. There should be only sea on the north of the coast.
![echo replicas tc](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040109_194924_000000182023_00157_09730_1479_echo_replicas_tc.png)

This occurs because of the nature of the radar acquisition mechanism, which is best described
in the [video](https://youtu.be/74p1uPCwU_c?si=mWoRadfr7bX2c2Rg) below

<iframe width="560" height="315" src="https://www.youtube.com/embed/74p1uPCwU_c?si=mWoRadfr7bX2c2Rg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

As can be seen that the single point on ground constitutes to data of many acquisitions e.g. lines/samples. In order to
focus a certain area it should be compressed over certain amount of pixels which can be loosely described
as "aperture pixels" e.g. instrument's area of acquisition.

In general the boundary areas in azimuth direction cannot be focussed correctly of the L0 dataset (same for range actually).
During the processing the lines near sensing start and/or end should be padded/preceded/prepended by extra lines because of these phenomena.
That's why there are always **more lines processed** than the actual level 1 product.

For example a reference PF-ASAR L1 IMS product (ASA_IMS_1PNESA20040109_194924_000000182023_00157_09730_0000.N1)
has the following dimensions\
`5174 x 30181`\
Metadata  - `num_lines_proc = 31514`\
So there are extra 1333 lines processed for correctly focussed picture.

Next chapters describe how this is implemented by the **asar-focus** processor.

# Implementation

Below is given the focussed product of another example dataset - ASA_IM__0PNPDK20040229_212454_000000492024_00387_10461_2585.N1\
It will be used throughout this chapter. It has a long sensing period total of 50 seconds, which gives enough packets
for padding for the correct results.
```
SENSING_START="29-FEB-2004 21:24:54.544118"
SENSING_STOP="29-FEB-2004 21:25:44.320391"
```

Processing L1 uses the following filter for start and end:
```xml
        <Sensing_Time>
            <Start>20040229_212504912000</Start>
            <Stop>20040229_212523002000</Stop>
        </Sensing_Time>
```

Without the extra lines the results are following (notice the artifacts at the bottom):\
![artifacts bottom](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040229_212454_000000492024_00387_10461_2585_artifacts_bottom.png)

While reference PF-ASAR results are following (ASA_IMS_1PNESA20040229_212504_000000172024_00387_10461_0000.N1):\
![IMS reference bottom](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNESA20040229_212504_000000172024_00387_10461_0000_bottom.png)

## Revised pipeline

In order to produce the correct product the parsing and measurements handling pipeline was reworked.
This part did not influence any compressing and signal processing logic. Also no drop in performance.

![windowing pipeline](https://github.com/kautlenbachs/bulpp_diagrams/blob/main/asar_meta_flow.drawio.png?raw=true)

This enabled the following:
* Collect enough packets to successfully produce results without artifacts
* Control and handle metadata and measurements for final cut and windowing
* Respect assigned sensing start and end times
* For datasets where sensing start/end does not enable padding, correct conditioning of results are still possible

Below are presented corrected results that match the reference

![windowed bottom](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040229_212454_000000492024_00387_10461_2585_windowed_bottom.png)

## Range cuts

Besides azimuth direction padding there are also some padded areas created by the processing algorithms.
Main ones are RCMC (Range cell migration correction) and SWST (sampling window start time) introduced near and far range
pixels that should be cut from the final result. They are either "nodata" resulting from shifting and RCMC windowing
algorithm or range compressed pixels which do not contain very well focussed pixels due to the nature of compressions.

Below is unfocused picture of an intermediate processing product which contains initial measurement samples with SWST
influenced shifts. It is the same product as been used before. Note the near and far range empty areas which is
influenced by SWST change. In this case SWST changed from 1532 to 1568 resulting in 36 padded pixels to near and
far range respectively.

![raw swst padding](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040229_212454_000000492024_00387_10461_2585_raw_swst_range_padding.png)

Before any of these effects of the processing were not compensated as illustrated by the excerpt below
(fully focussed result)

![range padding pixels](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040229_212454_000000492024_00387_10461_2585_near_range_padding_pixels.png)

The RCMC window size is 16 at the time of the writing and roughly there would be half of the pixels needed to be removed
from near and far range making it the total of the window size itself. On the picture above the gray pixels are the 
pure window operation padding picture (around 5-6 in this example) and another 2-3 pixels are not well focussed ones.
Below are results after correction. The scene starts and ends with correctly focussed pixels without any residue.

![near range padding removed pixels](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040229_212454_000000492024_00387_10461_2585_near_range_padding_removed_pixels.png)

Interesting findings were spotted on reference PF-ASAR processor. The same input product and resulting L1 focussed IMS
dataset seems to not correctly compensate for range/line focussing padded pixels. Dataset name - ASA_IMS_1PNESA20040229_212504_000000172024_00387_10461_0000.N1

![near range padding pixels reference](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNESA20040229_212504_000000172024_00387_10461_0000_near_range_padding_pixels.N1)

This is present on near and far range. One could conclude the following:
* SWST change might not be correctly handled
* If SWST change has been correctly handled then the area (around 36 pixels due to SWST difference) in near/far range has 
  not been correctly focussed as the pixel values are blurred along the section near SWST change, finally converging into properly focussed pixels along azimuth direction 
* Pixels placement/handling could be wrong for the processor currently developed and described in this post.

Same section of the same focussed L0 dataset is presented below by the GPU processor (note not so well focussed reflections - but this is a topic of another post)

![near range pixels comparison](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040229_212454_000000492024_00387_10461_2585_near_range_comparison.png)

Same scene, far range SWST change. Results displayed are terrain corrected and displayed on map in QGIS.
First reference processor, then GPU processor

![far range pixels comparison reference](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNESA20040229_212504_000000172024_00387_10461_0000_far_range_tc_swst_change_qgis.png)

![far range pixels comparison gpu processor](https://sar-focusing.s3.eu-central-1.amazonaws.com/pages/ASA_IMS_1PNPDK20040229_212454_000000492024_00387_10461_2585_far_range_tc_swst_change_qgis.png)