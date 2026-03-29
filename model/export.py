"""
Exports trained MiniVGG weights to a C header file.

Performs 8-bit quantization on weights and biases to prepare the model
for inference on FPGA or embedded devices.
"""

import torch
import torch.nn as nn
import os
import numpy as np

# Configuration
MODEL_PATH = 'minivgg.pth'
OUTPUT_HEADER = 'weights.h'
FRACTIONAL_BITS = 4
SCALING_FACTOR = 16

class MiniVGG(nn.Module):
    """Network definition used to load the saved state dictionary."""
    def __init__(self):
        super(MiniVGG, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Linear(64 * 4 * 4, 2) 
    def forward(self, x): return x

def quantize(value):
    """Scales float values and clamps them to 8-bit signed integers."""
    scaled = value * SCALING_FACTOR
    rounded = int(round(scaled))
    if rounded > 127: rounded = 127
    if rounded < -128: rounded = -128
    return rounded

def write_array_to_header(f, name, tensor):
    """Writes a flattened tensor to the file as a C array."""
    flat_data = tensor.detach().cpu().numpy().flatten()
    print(f"Exporting {name}: {len(flat_data)} elements...")
    f.write(f"// Layer: {name} (Shape: {list(tensor.shape)})\n")
    f.write(f"static const signed char {name}[{len(flat_data)}] = {{\n    ")
    for i, val in enumerate(flat_data):
        q_val = quantize(val)
        f.write(f"{q_val}, ")
        if (i + 1) % 16 == 0: f.write("\n    ")
    f.write("\n};\n\n")

def main():
    if not os.path.exists(MODEL_PATH): 
        print("Model not found!")
        return
    
    # Load trained model
    model = MiniVGG()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # Generate Header
    with open(OUTPUT_HEADER, 'w') as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        
        # Export Convolutional Layers
        write_array_to_header(f, "conv1_weights", model.layer1[0].weight)
        write_array_to_header(f, "conv1_bias",    model.layer1[0].bias)
        write_array_to_header(f, "conv2_weights", model.layer2[0].weight)
        write_array_to_header(f, "conv2_bias",    model.layer2[0].bias)
        write_array_to_header(f, "conv3_weights", model.layer3[0].weight)
        write_array_to_header(f, "conv3_bias",    model.layer3[0].bias)
        
        # Export Dense Layers
        write_array_to_header(f, "dense_weights", model.classifier.weight)
        write_array_to_header(f, "dense_bias",    model.classifier.bias)
        
        f.write("#endif\n")
    print(f"Standard weights restored to {OUTPUT_HEADER}")

if __name__ == "__main__":
    main()