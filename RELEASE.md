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