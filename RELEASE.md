# Release 0.2.0

## Breaking changes
* CLI arguments are defined and shall be used correctly in order to invoke processing

## Known Caveats
* 

## Major Features and Improvements
*

## Bug Fixes and Other Changes

## Thanks to our Contributors

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

Kajal Haria, Sabrina Pinori, Fabiano Costantini, Marco Galli from SERCO\
Andrea Recchia from aresys