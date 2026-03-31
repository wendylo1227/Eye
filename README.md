# Eye

## Overview
A real-time driver drowsiness detection system based on FPGA acceleration.  
This project detects whether a driver's eyes are open or closed by combining OpenCV-based ROI detection, DMA data transfer, and a Mini-VGG CNN accelerator implemented with Vivado HLS on the PYNQ platform.

## Features
- Real-time eye-state detection for driver drowsiness monitoring
- FPGA-accelerated Mini-VGG CNN inference
- OpenCV Haar Cascade for face and eye region detection
- DMA-based data transfer between processor and FPGA
- Visual warning system:
  - Green bounding box: eyes open
  - Red bounding box: eyes closed
  - Additional warning text when both eyes are closed

## Technologies
- C / C++
- Python
- FPGA / Vivado HLS
- PYNQ
- OpenCV
- AXI-Stream / DMA

## Project Structure
- `DEMO/` : demo files and project presentation materials
- `HLS/` : HLS-based CNN accelerator design
- `RTL/` : hardware design and block design related files
- `app/` : software-side application for system control and display
- `model/` : CNN training, preprocessing, and exported model parameters

## How to Run
1. Train the Mini-VGG CNN model with eye image datasets.
2. Export trained weights into fixed-point format for FPGA deployment.
3. Implement the CNN accelerator in Vivado HLS.
4. Build the hardware system in Vivado and generate the bitstream.
5. Load the bitstream on the PYNQ board.
6. Run the Python application to:
   - capture camera input
   - detect face and eye ROI using OpenCV
   - transfer ROI data through DMA
   - perform FPGA inference
   - display eye-state results in real time

## Result
![DEMO_picture](DEMO/DEMO_picture.jpg)
- Green frame: normal
- Red frame: alert
- When both eyes are closed, an additional warning message is shown

## Application
This project can be applied to real-time driver assistance and drowsiness monitoring systems.

## DEMO
https://drive.google.com/drive/folders/1xw2upyogDU1Ye17T1XFRkOC_ur9KVdm8?usp=drive_link
