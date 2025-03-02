# Algerian-products-detection

# Product Detection System

A full-stack computer vision solution for grocery product detection, combining YOLOv8 models with a web interface.

![Demo](demo-screenshot.jpg)

## Features
- **Dual-Model Ensemble** 
  - Combines predictions from two YOLOv8 models for improved accuracy
  - Intelligent IoU-based conflict resolution between models
- **Multi-Input Support**
  - Real-time video frame processing
  - Single image upload capability
- **Customizable Confidence** 
  - Adjustable detection threshold (0.1-0.9)
- **Visual Feedback**
  - Bounding boxes with confidence scores
  - Class labels from both models
- **API Endpoints**
  - `/detect_image` for image processing
  - `/detect_video` for video streams

## Tech Stack
- **Backend**: Python 3.11, Flask, Ultralytics YOLOv8
- **Frontend**: HTML5, JavaScript, Canvas API
- **Computer Vision**: OpenCV, NumPy
- **Optimization**: Multi-threaded inference, Base64 encoding

## Installation
1. Clone repository:
