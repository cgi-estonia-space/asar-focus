---
title: Version 0.3.x shortcomings
date: 2024-01-08 10:00:00 +0200
categories: [focussing, compression]
tags: [metadata,range,azimuth,chirp,doppler,noise,scaling]
---

# Background

The final focussing quality of the re-developed processor is not ideal. The results seem visually fine, but when analyzing
energy formation, sidelobes etc. they are off. Below are the main topics that the development team has pointed out. Some
of the approaches have been tested but with little to no information of the reference processor it is hard to determine
the root cause(s).

There are open issues/tasks in the GitHub repository as well (which are described in this document):
* [Focusing quality improvements](https://github.com/cgi-estonia-space/asar-focus/issues/2)
* [DAR implementation](https://github.com/cgi-estonia-space/asar-focus/issues/1)
* [Chirp replica](https://github.com/cgi-estonia-space/asar-focus/issues/7)
* [Calibration](https://github.com/cgi-estonia-space/asar-focus/issues/23)

## Secondary Range Compression

From "**Digital Processing of Synthetic Aperture Radar Data**":

![](/assets/rda_options.png)

The current version of the processor implements the "**Basic RDA**" variant.

The "**approximate SRC**" and "**accurate SRC**" were developed as described in `Cumming & Wong "Digital Processing of
Synthetic Aperture Radar Data"`. However, no improvement in focusing quality was measured. One of the causes could be
that implementation of the SRC missed some details or the test datasets did not have enough squint to matter -
hence it is not included in the `0.3.x` release. However, this could need further analyze if this could be one of the
reasons to match the focussing quality compared to the reference processor.

## Doppler ambiguity

Doppler centroid consists of two parts - **integer part** and **the fractional part**.
Current processor calculates and applies the fractional Doppler centroid, however it does not properly estimate or
apply the integer part of the Doppler Centroid.
Thus, if the antenna is squinted too much, and the azimuth Doppler history does not lay within **+-0.5PRF**
(Pulse Repetition Frequency), it would cause aliasing.

Below is an image from article ["**Signal Properties of Spaceborne Squint-Mode SAR**"](https://www.researchgate.net/publication/3201663_Signal_Properties_of_Spaceborne_Squint-Mode_SAR).
It visualizes the situation well - if the antenna is not ideally side-looking, the observed Doppler frequencies may not
cross the zero point.

![](/assets/squint.png)

For example, if the PRF is **2000Hz**, the Doppler centroid is estimated correctly in the **+-1000Hz** range because the
current processor only calculates the fractional part, however if the true Doppler centroid would be **1500Hz**,
it would be estimated as **500Hz** due to aliasing and no integer part would be estimated. This can be observed by
invoking the processor with `--plot` argument. This would output some metrics like Doppler centroid in a separate dataset.
However, this should be more of an issue with ERS datasets, where the true Doppler centroid is outside the **+-0.5 PRF**.

# Average terrain height

The 0.3.x version processor always assumes the scene height from WGS84 ellipsoid is 0 meters. This could cause two
issues when dataset is not at the sea level:
* degraded geocoding
* worsened estimate of **Vr**(processing velocity/radar velocity) which is used during azimuth compression.

Currently it is implemented as following:

1. Geocoding is done via finding a Zero-Doppler point from orbit state vectors:
![](/assets/geocoding.png)
2. **Vr** is estimated from orbit state vector data, as described in the "Digital processing of Synthetic aperture data" chapter 13.3.
![](/assets/vr_estimate.png)


Both operations involve estimating the height of the target point, however it is unsure if this should require
DEM (Digital Elevation Model) to estimate the center scene height more precisely as there has been mixed feedback from
experts.

Nevertheless, the lack of average terrain height estimate does not cause poorer focussing quality on sea level datasets
that has been observed.

# Chirp replica generation

The current processor utilizes the nominal range chirp to generate the reference chirp and uses it to perform range
compression.

An example nominal chirp looks like this for ASAR:
![](/assets/nom_chirp.png)

The metadata contained in the (auxiliary) dataset(s) could be used to replicate the chirp from
measurements. Utilizing it in range compression could in turn improve the azimuth compression results. Again, it is
difficult to know in advance how much difference it would make.

# Azimuth windowing

Based on the current analyze and knowledge it is assumed that the processor correctly performs windowing for the
range compression.

A window is applied before range compression, the reference chirp above is then transformed as seen below:
![](/assets/window_chirp.png)

This is a standard DSP technique for pulse compression. The equivalent could be applied for
azimuth compression as well. However, due to the complexity of the azimuth compression (Doppler centroid,
applying in frequency domain, rather than time domain) the implementation is more complicated.

Due to this the current version of the processor does not apply azimuth windowing.
Tests with a potential implementation were made, however the results did not show clear improvements,
unlike in range direction where results improved after windowing.

Again, this could have been due to missing details in the implementation. Finally, this further complicated the scaling
issue and it is not currently included.

# Azimuth block processing

After the range compression, the intensity image of a range compressed image is visualized below (where x-axis is range and y-axis azimuth direction):

![](/assets/rc.png)


The core part of Range Doppler Algorithm is to perform azimuth compression in range-time and azimuth frequency domain
and for this azimuth FFT is taken of the signal data. The current processor version computes the azimuth direction FFT
in one block, meaning the FFT size is `azimuth size + FFT padding`. This approximation does not adjust the 
**Vr** (radar velocity) in azimuth direction. More precise approach would divide the azimuth into blocks
and take multiple smaller FFTs that cover the azimuth aperture to utilize individual **Vr** estimate for each block.

Experiments with the **Vr** estimation reference at other points than the center point and additionally performing
azimuth compression only on a subset of the product to allow **Vr** to better match did not produce
clear-cut focusing quality improvements - thus it is not clear if this could enhance the results.

## Kaiser window

The range window always defaults to Hamming window, ignoring the processor configuration auxiliary file.
Since it was observed that the processor configuration files often stated "HAMMING", this should have minimal effect.

# Metadata

There are many issues/tasks listed in the repository:
* [Metadata fixes](https://github.com/cgi-estonia-space/asar-focus/issues/10)
* [SQ ADS](https://github.com/cgi-estonia-space/asar-focus/issues/24)
* [Implement metadata placeholders](https://github.com/cgi-estonia-space/asar-focus/issues/40)
* [Processing facility](https://github.com/cgi-estonia-space/asar-focus/issues/21)

Main reasons are time and/or information shortage. Most of the missing fields are quick to implement if there would be
detailed specification. Most of the current metadata was assembled together by consulting available documentation and
manually reviewing (auxiliary) datasets. However, to match the reference processor more information is required. Some 
examples:
* `PRODUCT_ERR` field - there could be many sources like invalid SWST(s), missing counters, packet gaps etc.
* `downlink_flag` of the SQ ADS
* `VECTOR SOURCE` - no map for the acronyms

Also, the root causes vary between ERS and ENVISAT missions. Some statistics is gathered by the processor internally 
like data record number gaps, out of sequence counters etc. but this is output to the console only.

## Scaling (Calibration constant;external calibration scaling factor)

The processor's final output for IMS and IMP mode pixels are scaled by essentially magic constants that was found to
produce best match when compared to reference datasets:
```
        const auto swath_idx = alus::asar::envformat::SwathIdx(asar_meta.swath);
        auto tambov = conf_file.process_gain_ims[swath_idx] / sqrt(xca.scaling_factor_im_slc_vv[swath_idx]);
        if (instrument_type == alus::asar::specification::Instrument::SAR) {
            tambov /= (2*4);
        }
```

To the best knowledge, the intensity sum of data processor stays relatively similar from
`RAW -> Range compression -> Azimuth Compression -> Final image` so it should just be a question of how the final pixel
values are converted from floating point numbers to 16-bit integers in the final product.
However, no clear answer of how this is performed and how the metadata in the configuration files are used. Also, it is 
stated that this varies between processing facilities.

Read more in the [GitHub issue](https://github.com/cgi-estonia-space/asar-focus/issues/23)

## IMP mode

The IMP mode multilooking is implemented via time domain averaging (similar algorithm to MultilookOp.java in ESA SNAP)
and it is then linearly resampled the azimuth direction to get the 12.5m pixel spacing.

The correct way to implement this would be after azimuth compression, but before backwards Azimuth FFT.
The implementation should IFFTs from part of the spectrums and resample at the same time.
This would be called the "Frequency domain" implementation, however this could not be implemented properly.
Also, it is unclear if this would improve the current "Time domain" implementation.
