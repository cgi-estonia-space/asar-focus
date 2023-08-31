#pragma once

#include <cstdint>


/**
 * ASAR specific constants use for converting raw data to physical values
 */

// PO-TN-MMS-SR-0248 - conversion tables
float UpconverterLevelConversion(uint8_t up);

float AuxTxMonitorConversion(uint16_t aux_tx);
