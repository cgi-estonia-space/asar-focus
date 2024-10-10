---
title: Version 0.2 shortcomings
date: 2024-01-08 10:00:00 +0200
categories: [focussing, compression]
tags: [metadata,range,azimuth,chirp,doppler,noise,scaling]
---

# Background

TODO?

# Focusing quality

The asar-focus focusing quality is not ideal, we considered many possible improvements that we either tested or could not implement.

## Secondary Range Compression

From "Digital Processing of Synthetic Aperture Radar Data"

![](/assets/rda_options.png)

The current version of the processor implements the "Basic RDA" variant.

Implementation attempts were made of the "approximate SRC" and "accurate SRC" as described in Cumming&Wong "Digital Processing of
Synthetic Aperture Radar Data", however we could not find an improvement in focusing quality. Perhaps the SRC implemenation was incorrect or the datasets we tested with did not have enough squint to matter - either way we left it out 0.2 release. However this needs further study if this could be one of the reasons for poorer focussing quality compared to the reference processor.

## Doppler ambiguity


Doppler centroid is comprised of two parts - integer part and the fractional part.
The current processor calculates and applies the fractional Doppler centroid, however it does not properly estimate or apply the integer part of the Doppler Centroid.

Thus if the antenna is squinted too much, and the azimuth Doppler history does not lay withing +- 0.5PRF, it will be aliased.

Image from article - Signal Properties of Spaceborne Squint-Mode SAR, visualizing how this happens - if the antenna is not ideally sidelooking, the observed Doppler frequencies may not cross the zero point.

![https://www.researchgate.net/publication/3201663_Signal_Properties_of_Spaceborne_Squint-Mode_SAR](/assets/squint.png)

For example, if PRF is 2000HZ, the Doppler centroid is estimated correctly in the +-1000Hz range because the current processor only calculates the fractional part, however if the true Doppler centroid is for example 1500Hz, it would be estimated as 500Hz due to aliasing and no integer part would be estimated. The Doppler centroid DS in the output file would display incorrectly and the focusing quality would be degraded.

This is should be more of an issue with some ERS datasets, where the true Doppler centroid is outside of +-0.5 PRF.

# Average terrain height

The 0.2 processor always assumes the dataset height from WGS84 ellipsoid is 0m. This should cause two immediate issues with dataset not at the sea level - 1. degraded geocoding 2. worse estimate of Vr(processing velocity/radar velocity) used for calculation during azimuth compression.

1. Geocoding is done via finding an zero Doppler point from orbit state vectors, visually something like this:
![](/assets/geocoding.png)
2. Vr is estimated from orbit state vector data, as described in the "Digital processing of Synthetic aperture data" in 13.3.
![](/assets/vr_estimate.png)


Both operations involve estimating the height of the target point, however we were unsure of the implementation for this - does other focussing software require DEM access to estimate the center scene height.

Nevertheless, the lack of average terrain height estimate does not explain poorer focussing quality on sea level datasets.

# Chirp replica generation

The current processor always uses the nominal range chip to generate the reference chirp and use it to perform range compression.

An example nominal chirp look like this for ASAR:
![](/assets/nom_chirp.png)

Potentially using the metadata contained the dataset could be used the replicate the chirp from measurements. This could be used in range compression and this in turn improve the azimuth compression result. However, it is difficult for us to known in advance, how much of a difference it makes in practise.

# Azimuth windowing

The processor correctly(?) performs windowing for the range compression.

An window is applied before range compression, the previous reference chip look like this after the window(but before scaling)
![](/assets/window_chirp.png)

This windowing is a standard DSP technique for pulse compression. The equivalent should be applied for azimuth compression. However due to the details of the of the azimuth compression(Doppler centroid, applying in frequency domain, rather than time domain) the implementation is more complicated.

Due to this the current version of the processor does not apply azimuth windowing. Internal test with a potential implementation were made, however the results were not a clear improvements, unlike in range direction where the window is implemented in range compression. Again this could have been due to an incorrect implementation, as this further complicated the scaling issue, it was left out.


# Azimuth block processing


After range compression, the intensity image of a range compressed image look something like this, with x-axis being range and y-axis azimuth direction.

![](/assets/rc.png)


The core part of Range Doppler Algorithm is to perform azimuth compression in range-time and azimuth frequency domain and for this azimuth FFT is taken of this signal data. The current processor version always does the azimuth direction FFT in one block, meaning the FFT size is azimuth size + FFT padding. This simplifies implementation, however does not allow the implementation to vary Vr(radar velocity) in azimuth direction. An better approach would be to divide the azimuth into blocks, and take multiple smaller FFT that cover the azimuth aperture, to allow individual Vr estimate for each block.

Internal tests with taking the Vr estimation reference at other points than the center point and additionally performing azimuth compression only on a subset of the product, to allow for Vr to better match the did not produce clear-cut focusing quality improvements, thus it is not clear ot us if this is the problem.


## Kaiser window
The range window always default to Hamming window, ignoring the processor configuration. This should have minimal effect?



# Metadata

TODO

# Misc

## IMP mode

The IMP mode multilooking is implemented via time domain averaging(very similar to MultilookOp.java in ESA SNAP) and it is then linearly resampled the azimuth direction to get the 12.5m pixel spacing.

The correct way to implement this would be after azimuth compression, but before backwards Azimuth FFT. The implementation should IFFTs from part of the spectrums and resample at the same time. This would be called the "Frequency domain" implementation, however we could not manage to implement it properly. Additionally it unclear to us if this would improve on the current "Time domain" implementation.

## Scaling

The processor's final output for IMS and IMP mode pixels are scaled by essentially magic constants that we found to produce close enough results for the reference datasets.

To the best of our knowledge, the intensity sum of data processor stays relatively similar from RAW -> Range compression -> Azimuth Compression -> Final image, at it should just be a question of how the final pixel values are converted from floating point numbers to 16 integers in the final product. However we could not find a clear answer of this is peformed and how the metadata in the configuration files is ultimately used.