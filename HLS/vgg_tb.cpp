/**
 * @file vgg_tb.cpp
 * @brief Testbench for the Mini-VGG FPGA Accelerator.
 * * Generates a synthetic 32x32 image pattern, streams it to the accelerator,
 * and verifies that the output matches the expected format (2 class probabilities).
 */

#include <iostream>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "definitions.h"

// Top-level function prototype
void vgg_accelerator(hls::stream<axis_t> &INPUT_STREAM, hls::stream<axis_t> &OUTPUT_STREAM);

int main() {
    std::cout << "--- Starting Mini-VGG Accelerator Testbench ---" << std::endl;

    // 1. Define AXI Streams
    hls::stream<axis_t> input_stream("input_stream");
    hls::stream<axis_t> output_stream("output_stream");

    // 2. Generate Test Data
    // Generates a 32x32 (1024 pixels) grayscale gradient pattern.
    std::cout << "[TB] Generating 32x32 input image..." << std::endl;
    
    for (int i = 0; i < 1024; i++) {
        axis_t pixel_packet;
        
        unsigned char pixel_val = (unsigned char)(i % 255);
        
        // Configure AXI Side-Channels
        pixel_packet.data = pixel_val;   // 8-bit Pixel data
        pixel_packet.keep = 0xF;         // Keep all bytes
        pixel_packet.strb = 0xF;
        pixel_packet.user = 0;
        pixel_packet.id   = 0;
        pixel_packet.dest = 0;
        
        // Assert TLAST on the final pixel of the frame
        pixel_packet.last = (i == 1023) ? 1 : 0;

        input_stream.write(pixel_packet);
    }

    // 3. Execute Hardware Logic
    std::cout << "[TB] Invoking vgg_accelerator..." << std::endl;
    vgg_accelerator(input_stream, output_stream);

    // 4. Verify Output
    // Expecting 2 output packets (Class 0 and Class 1) in Fixed Point <8,4> format.
    std::cout << "[TB] Reading outputs..." << std::endl;

    if (output_stream.empty()) {
        std::cerr << "[ERROR] Output stream is empty! Accelerator failed." << std::endl;
        return 1;
    }

    int packet_count = 0;
    while (!output_stream.empty()) {
        axis_t out_packet = output_stream.read();
        
        // Decode Fixed Point <8,4> to Float for verification
        signed char raw_val = (signed char)(out_packet.data & 0xFF);
        float real_val = (float)raw_val / 16.0f; 

        std::cout << " -> Output Class " << packet_count << ": Raw=" << (int)raw_val 
                  << " Fixed=" << real_val << " Last=" << (int)out_packet.last << std::endl;
        
        packet_count++;
    }

    // 5. Final assertion
    if (packet_count != 2) {
        std::cerr << "[FAIL] Expected 2 output packets, got " << packet_count << std::endl;
        return 1;
    }

    std::cout << "--- Testbench Completed Successfully ---" << std::endl;
    return 0;
}