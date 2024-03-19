# Release 0.3.0

## Breaking changes

## Known Caveats
* Only IMS and IMP product processing
* Azimuth compression windowing is yet to be done - https://github.com/cgi-estonia-space/asar-focus/issues/2
* Processing speed (Vr) and Doppler centroid changes in azimuth direction yet to be done - https://github.com/cgi-estonia-space/asar-focus/issues/2
* Packets' ISP sensing time handling might not be 100% correct
  * It is observed for the reference products that it calculates new ISP sensing times based on PRI
  * Therefore products by this processor differ in sensing start/stop and first/last line times (always inside the specified sensing filter)
  * Best knowledge/effort basis changes has been implemented - https://github.com/cgi-estonia-space/asar-focus/issues/17 and https://github.com/cgi-estonia-space/asar-focus/issues/16
* Various metadata fields needs further work (Some were adressed during this release)
* Final results' scaling is yet to be determined, currently it is not matching exactly the reference processor
  * With the current experience/knowledge there is a "best guess" implemented

## Major Features and Improvements
* ERS time is corrected according to PATC files - https://github.com/cgi-estonia-space/asar-focus/issues/15
* IMP mode support along with metadata enhancements - https://github.com/cgi-estonia-space/asar-focus/issues/4
  * Ground detected metadata created
  * azimuth spacing, chirp ADS corrected

## Bug Fixes and Other Changes
* `SOFTWARE_VER` in the dataset changed to `asar_gpu/M.m.p`

## Thanks to our Contributors

Kajal Haria, Fabiano Costantini from Telespazio UK\
Sabrina Pinori, Marco Galli from SERCO\
Andrea Recchia from aresys

# Release 0.2.0

## Breaking changes

## Known Caveats
* Only IMS product processing
* Not all the auxiliary files are utilized, currently processor configuration file (CON),
  instrument characterization file (INS) and external calibration data (XCA) are in use
* Azimuth compression windowing is yet to be done - https://github.com/cgi-estonia-space/asar-focus/issues/2
* Processing speed (Vr) and Doppler centroid changes in azimuth direction yet to be done - https://github.com/cgi-estonia-space/asar-focus/issues/2
* Packets' ISP sensing time handling might not be 100% correct
  * It is observed for the reference products that it calculates new ISP sensing times based on PRI
  * Therefore products by this processor differ in sensing start/stop and first/last line times (always inside the specified sensing filter)
  * Best knowledge/effort basis changes has been implemented - https://github.com/cgi-estonia-space/asar-focus/issues/17 and https://github.com/cgi-estonia-space/asar-focus/issues/16
* ERS time is not corrected according to PATM/N/C files - https://github.com/cgi-estonia-space/asar-focus/issues/15
* Various metadata fields needs further work
* Final results' scaling is yet to be determined, currently it is not matching exactly the reference processor
  * With the current experience/knowledge there is a "best guess" implemented
* See more shortcomings at https://cgi-estonia-space.github.io/asar-focus/posts/version-0-2-shortcomings/

## Major Features and Improvements

* Range and azimuth direction edge pixels are cut from the final result, [read more](https://cgi-estonia-space.github.io/asar-focus/posts/range-and-azimuth-window/)
* CUDA device initialization, properties and check util
* Storing results made faster with async IMS write and higher GPU utilization -
  see https://github.com/cgi-estonia-space/asar-focus/pull/26 and https://cgi-estonia-space.github.io/asar-focus/posts/read-and-write-optimizations/
* Complete refactor of parsing the input files, fetching measurements and processing - almost 3x of processing time 
  gains - https://github.com/cgi-estonia-space/asar-focus/pull/29 See more detailed description - https://cgi-estonia-space.github.io/asar-focus/posts/read-and-write-optimizations/
* Better handling of missing packets and various fields for handling the missing measurements etc
* Usage of external calibration file for scaling factor guess - https://github.com/cgi-estonia-space/asar-focus/pull/34
* Final results without artifacts and 'nodata' pixels - see https://github.com/cgi-estonia-space/asar-focus/pull/33 and https://cgi-estonia-space.github.io/asar-focus/posts/range-and-azimuth-window/
* Hamming window applied for range compression - https://github.com/cgi-estonia-space/asar-focus/pull/34
* More metadata fields computed/assembled

## Bug Fixes and Other Changes
* Invalid read of single additional sample of Envisat measurement fixed - https://github.com/cgi-estonia-space/asar-focus/pull/30
* Memory leak of the compressed sample block fixed - https://cgi-estonia-space.github.io/asar-focus/posts/read-and-write-optimizations/
* Additional output for plot (`--plot`) that enables to analyze the geolocation of the scene and basic parameters
* Fixed azimuth direction artifacts with padded lines for compression -
  see https://github.com/cgi-estonia-space/asar-focus/pull/33 and https://cgi-estonia-space.github.io/asar-focus/posts/range-and-azimuth-window/
* Envisat results scene geolocation inaccuracy fixed by applying gate bias/delay in the calculation - https://github.com/cgi-estonia-space/asar-focus/pull/33
* More modules formed as CMake targets with unit tests

## Thanks to our Contributors

Kajal Haria, Fabiano Costantini from Telespazio UK\
Sabrina Pinori, Marco Galli from SERCO\
Andrea Recchia from aresys

# Release 0.1.2

## Breaking changes
* CLI arguments are defined and shall be used correctly in order to invoke processing, no more positional arguments

## Known Caveats
* Only IMS product processing
* Not all the auxiliary files are used/supported, only instrument (INS) and configuration (CON)
* Range and azimuth compression windowing is yet to be done - https://github.com/cgi-estonia-space/asar-focus/issues/2
* Processing speed (Vr) and Doppler centroid changes in azimuth direction yet to be done - https://github.com/cgi-estonia-space/asar-focus/issues/2
  * This means that the focussing quality is not exactly on par with the reference processor
* Packets' ISP sensing time handling might not be 100% correct
  * It is observed for the reference products that it calculates new ISP sensing times based on PRI
  * Therefore products by this processor differ in sensing start/stop and first/last line times (always inside the specified sensing filter)
  * Best knowledge/effort basis changes has been implemented - https://github.com/cgi-estonia-space/asar-focus/issues/17 and https://github.com/cgi-estonia-space/asar-focus/issues/16
* ERS time is not corrected according to PATM/N/C files - https://github.com/cgi-estonia-space/asar-focus/issues/15
* Metadata for SQ_ADS, CHIRP_PARAM_ADS, etc. is not constructed currently - https://github.com/cgi-estonia-space/asar-focus/issues/10
  * Other non DSD specific metadata as well - LEAP_SIGN, LEAP_ERR, ORBIT etc...
* Final results' calibration constant is yet to be determined, currently it is not matching exactly the reference processor
  * With the current experience/knowledge it must be experimented and tested, no known specific formula for it

## Major Features and Improvements
* Sensing start and end arguments
* ERS and ENVISAT missing packets insertion
* For ENVISAT all calibration packets are inserted with echo packet
* Different modes/products generation is tracked and handled accordingly
* Less explicit exits, everything is routed through exceptions with descriptive messages
* Proper log system setup using Boost log, with 5 levels. Can be set when invoking the processor
* Aux folder handling more robust - can identify aux files implicitly
* Can parse proper orbit file out of the folder with miscellaneous aux files (TDS sets)
* Echo metadata is ERS and ENVISAT specific, more fields parsed and used during processing (packet counters for example)
* Diagnostic metric by default enabled, they are printed out after parsing step (different counters out of sequence and gap metrics)
* Minor metadata improvements regarding processor, ref doc, compression method
* FEP annotations' MJD for ENVISAT is parsed correctly
* Faster parsing/preparing of echos from input dataset, from 1++ second to ~0.7 second

## Bug Fixes and Other Changes
* Output file `FILE` handle check and other undefined states now guarded
* Half of the modules are now CMake modules with unit tests
* Can be compiled with clang (version 14 tested) as well, compiler flags and build improvements executed for this

## Thanks to our Contributors

Kajal Haria, Fabiano Costantini from Telespazio UK\
Sabrina Pinori, Marco Galli from SERCO\
Andrea Recchia from aresys

# Release 0.1.1

## Breaking changes

## Known Caveats

## Major Features and Improvements
* Added CUDA architecture support from 5.0 to 8.0 for the released binaries

## Bug Fixes and Other Changes

## Thanks to our Contributors


# Release 0.1.0

## Breaking changes

## Known Caveats
* Final calibration of the foccussed pixels is not done in order to adhere to the baseline processor
* ERS and ENVISAT packets are not filtered correctly - i.e. by sensing start/end or broken ones - all of the packets are included for processing
* Minor pixel offsets/shifts are present
* No PATC/PATN information is included for ERS on board time correction
* Residue blocks are not processed for ENVISAT, only the whole 64 chunks are (range size a little smaller)

## Major Features and Improvements
* Initial version with ERS and ENVISAT IM support with only IMS target
* Doris Orbit files are supported
* Auxiliary files support for configuration and instrument dataset

## Bug Fixes and Other Changes

## Thanks to our Contributors

Kajal Haria, Fabiano Costantini from Telespazio UK\
Sabrina Pinori, Marco Galli from SERCO\
Andrea Recchia from aresys
