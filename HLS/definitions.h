/**
 * @file definitions.h
 * @brief Global type definitions and constants for the HLS design.
 * * Defines the fixed-point precision levels used for feature maps, 
 * weights, and accumulation registers to ensure timing closure and accuracy.
 */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <ap_axi_sdata.h>
#include <ap_fixed.h>

// --- DIMENSIONS ---
#define IMG_H 32
#define IMG_W 32

// --- DATA TYPES ---

// 1. Feature Maps (Activations)
// Precision: 8 bits total, 4 integer bits. Range: [-8.0, 7.9375]
typedef ap_fixed<8, 4, AP_RND, AP_SAT> fm_t;

// 2. Weights
// Precision: 8 bits total, 4 integer bits.
typedef ap_fixed<8, 4, AP_RND, AP_SAT> wt_t;

// 3. Accumulator
// Precision: 32 bits total, 16 integer bits.
// High precision required to prevent overflow during Dense Layer summation (1024 elements).
typedef ap_fixed<32, 16, AP_RND, AP_SAT> acc_t;

// 4. AXI Stream Interface
// Standard 32-bit AXI4-Stream packet.
typedef ap_axiu<32, 0, 0, 0> axis_t;

#endif