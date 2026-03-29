/**
 * @file vgg_accelerator.cpp
 * @brief Main HLS implementation of the Mini-VGG architecture.
 * * Implements a streaming CNN architecture using sliding window buffers 
 * for Convolution and Max Pooling, followed by a fully connected layer.
 */

#include <hls_stream.h>
#include "definitions.h"
#include "weights.h"

// Helper: Cast raw byte to Fixed Point Weight type
inline wt_t raw_to_fixed(signed char val) {
    wt_t res;
    res.range(7, 0) = val; 
    return res;
}

// Helper: Return maximum of two fixed point values
inline fm_t max_val(fm_t a, fm_t b) {
    return (a > b) ? a : b;
}

/**
 * @brief Generic Convolution Layer with Line Buffers.
 * * Implements a 3x3 convolution using a Line Buffer and Sliding Window approach.
 * This eliminates the need to store full frames in memory.
 * * @tparam IN_CH   Number of input channels.
 * @tparam OUT_CH  Number of output channels (filters).
 * @tparam IN_DIM  Spatial dimension (Width/Height) of the input.
 */
template<int IN_CH, int OUT_CH, int IN_DIM>
void layer_block(
    hls::stream<fm_t> &in_stream,
    hls::stream<fm_t> &out_stream,
    const signed char weights[OUT_CH][IN_CH][3][3],
    const signed char biases[OUT_CH]
) {
    // Line Buffer: Stores 2 rows of input data to support 3x3 window
    static fm_t line_buff[IN_CH][2][IN_DIM];
    #pragma HLS ARRAY_PARTITION variable=line_buff complete dim=2

    // Sliding Window: Stores 3x3 pixels for the current operation
    fm_t window[IN_CH][3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // Initialize line buffer
    for(int c=0; c<IN_CH; c++)
        for(int h=0; h<2; h++)
            for(int w=0; w<IN_DIM; w++) 
                line_buff[c][h][w] = 0;

    // Iterate over spatial dimensions
    for (int r = 0; r < IN_DIM; r++) {
        for (int c = 0; c < IN_DIM; c++) {
            
            //             
            // 1. Update Line Buffer & Window (II=1 for throughput)
            for (int ch = 0; ch < IN_CH; ch++) {
            #pragma HLS PIPELINE II=1
                
                fm_t val_in = in_stream.read();
                
                // Shift window columns left
                window[ch][0][0] = window[ch][0][1];
                window[ch][0][1] = window[ch][0][2];
                window[ch][1][0] = window[ch][1][1];
                window[ch][1][1] = window[ch][1][2];
                window[ch][2][0] = window[ch][2][1];
                window[ch][2][1] = window[ch][2][2];

                // Update Line Buffers
                fm_t val_row0 = line_buff[ch][0][c]; 
                fm_t val_row1 = line_buff[ch][1][c]; 
                
                line_buff[ch][0][c] = val_row1;
                line_buff[ch][1][c] = val_in; 
                
                // Fill new window column
                window[ch][0][2] = val_row0;
                window[ch][1][2] = val_row1;
                window[ch][2][2] = val_in;
            }

            // 2. Compute Convolution
            if (r >= 0 && c >= 0) { 
                for (int f = 0; f < OUT_CH; f++) {
                    
                    acc_t acc = raw_to_fixed(biases[f]);
                    
                    // Relaxed Pipeline (II=2) used here to meet timing constraints
                    // on the multi-channel accumulator chain (10ns period).
                    for (int ch = 0; ch < IN_CH; ch++) {
                    #pragma HLS PIPELINE II=2
                        
                        for (int wr = 0; wr < 3; wr++) {
                            for (int wc = 0; wc < 3; wc++) {
                                wt_t w = raw_to_fixed(weights[f][ch][wr][wc]);
                                fm_t px = window[ch][wr][wc];
                                acc += px * w;
                            }
                        }
                    }
                    // ReLU Activation
                    if (acc < 0) acc = 0; 
                    out_stream.write((fm_t)acc);
                }
            }
        }
    }
}

/**
 * @brief Max Pooling Layer (2x2).
 * * Downsamples the input by taking the maximum value in 2x2 windows.
 * Reduces spatial dimensions by half.
 */
template<int CH, int DIM>
void max_pool(
    hls::stream<fm_t> &in_stream,
    hls::stream<fm_t> &out_stream
) {
    static fm_t pool_buff[CH][DIM];
    #pragma HLS ARRAY_PARTITION variable=pool_buff complete dim=1

    fm_t window[CH][2][2];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < DIM; r++) {
        for (int c = 0; c < DIM; c++) {
        // Relaxed Pipeline (II=2) to allow sufficient time for comparison logic
        #pragma HLS PIPELINE II=2
            for (int ch = 0; ch < CH; ch++) {
                fm_t val_in = in_stream.read();

                // Shift Window
                window[ch][0][0] = window[ch][0][1];
                window[ch][1][0] = window[ch][1][1];

                // Update Line Buffer
                fm_t val_prev_row = pool_buff[ch][c];
                pool_buff[ch][c] = val_in; 
                window[ch][0][1] = val_prev_row; 
                window[ch][1][1] = val_in;       

                // Output valid result only on odd rows/cols
                if ((r % 2 == 1) && (c % 2 == 1)) {
                    fm_t max_0 = max_val(window[ch][0][0], window[ch][0][1]);
                    fm_t max_1 = max_val(window[ch][1][0], window[ch][1][1]);
                    out_stream.write(max_val(max_0, max_1));
                }
            }
        }
    }
}

/**
 * @brief Fully Connected (Dense) Layer.
 * * Flattens the input stream and computes the dot product for class scores.
 * Outputs the result via AXI Stream with TLAST asserted on the final class.
 */
void dense_layer(
    hls::stream<fm_t> &in_stream,
    hls::stream<axis_t> &out_stream,
    const signed char weights[2][1024],
    const signed char biases[2]
) {
    // Buffer entire flattened image (1024 elements)
    fm_t flat_img[1024];
    for (int i = 0; i < 1024; i++) {
    #pragma HLS PIPELINE II=1
        flat_img[i] = in_stream.read();
    }

    // Compute scores for 2 classes
    for (int c = 0; c < 2; c++) {
        acc_t acc = raw_to_fixed(biases[c]);
        
        // Relaxed Pipeline (II=2) to prevent adder chain violations
        for (int i = 0; i < 1024; i++) {
        #pragma HLS PIPELINE II=2
            unsigned int spatial_bits = i & 0xF; 
            unsigned int channel_bits = i >> 4;   
            unsigned int k = (spatial_bits << 6) | channel_bits;
            
            wt_t w = raw_to_fixed(weights[c][i]);
            acc += flat_img[k] * w; 
        }

        // Format AXI Output Packet
        axis_t output_packet;
        fm_t final_res = (fm_t)acc;
        output_packet.data = (int)final_res.range(7, 0);
        output_packet.keep = 0xF;
        output_packet.strb = 0xF;
        output_packet.last = (c == 1) ? 1 : 0;
        out_stream.write(output_packet);
    }
}

/**
 * @brief Input Adapter.
 * * Converts standard 8-bit integer AXI stream input to internal Fixed Point <8,4> format.
 */
void input_adapter(hls::stream<axis_t> &in_axi, hls::stream<fm_t> &out_internal) {
    for (int i = 0; i < 32 * 32; i++) {
    #pragma HLS PIPELINE II=1
        axis_t packet = in_axi.read();
        unsigned char raw = packet.data & 0xFF;
        fm_t val;
        // Simple scaling: Map 0..255 integer to Fixed Point
        unsigned char rounded = (raw + 8); 
        val.range(7, 0) = (rounded >> 4); 
        out_internal.write(val);
    }
}

/**
 * @brief Top-Level Hardware Function.
 * * Wires the layers together using HLS Dataflow to allow task-level parallelism.
 */
void vgg_accelerator(hls::stream<axis_t> &INPUT_STREAM, hls::stream<axis_t> &OUTPUT_STREAM) {
    // AXI Interfaces
    #pragma HLS INTERFACE axis port=INPUT_STREAM
    #pragma HLS INTERFACE axis port=OUTPUT_STREAM
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    // Inter-layer Streams (FIFOs)
    static hls::stream<fm_t> s_input("s_input");
    #pragma HLS STREAM variable=s_input depth=128
    
    static hls::stream<fm_t> s_conv1("s_conv1");
    #pragma HLS STREAM variable=s_conv1 depth=128
    static hls::stream<fm_t> s_pool1("s_pool1");
    #pragma HLS STREAM variable=s_pool1 depth=128
    
    static hls::stream<fm_t> s_conv2("s_conv2");
    #pragma HLS STREAM variable=s_conv2 depth=64
    static hls::stream<fm_t> s_pool2("s_pool2");
    #pragma HLS STREAM variable=s_pool2 depth=64
    
    static hls::stream<fm_t> s_conv3("s_conv3");
    #pragma HLS STREAM variable=s_conv3 depth=64
    static hls::stream<fm_t> s_pool3("s_pool3");
    #pragma HLS STREAM variable=s_pool3 depth=64

    // Dataflow Region: Enables concurrent execution of layers
    #pragma HLS DATAFLOW

    input_adapter(INPUT_STREAM, s_input);

    // Layer 1
    layer_block<1, 16, 32>(s_input, s_conv1, (const signed char(*)[1][3][3])conv1_weights, conv1_bias);
    max_pool<16, 32>(s_conv1, s_pool1);
    
    // Layer 2
    layer_block<16, 32, 16>(s_pool1, s_conv2, (const signed char(*)[16][3][3])conv2_weights, conv2_bias);
    max_pool<32, 16>(s_conv2, s_pool2);
    
    // Layer 3
    layer_block<32, 64, 8>(s_pool2, s_conv3, (const signed char(*)[32][3][3])conv3_weights, conv3_bias);
    max_pool<64, 8>(s_conv3, s_pool3);

    // Output Layer
    dense_layer(s_pool3, OUTPUT_STREAM, (const signed char(*)[1024])dense_weights, dense_bias);
}