# Sign Language Recognition with CNN

A PyTorch-based convolutional neural network for real-time American Sign Language (ASL) recognition using webcam input.

## Overview

This project implements a CNN model that recognizes 29 different ASL signs: 26 letters (A-Z) plus "space", "delete", and "nothing" gestures. The model processes 64x64 RGB images and provides real-time predictions through webcam feed.

## Requirements

Install dependencies using:

```
install -r requirements.txt
```

## Model Architecture

The CNN consists of:

- 3 convolutional layers (32, 64, 128 filters)
- MaxPooling after each conv layer
- 2 fully connected layers (512, 29 outputs)
- ReLU activations
- Input: 64x64 RGB images
- Output: 29 classes

## Usage

### 1. Data Preparation

If your test images aren't organized by class, run:

```bash
cd src/utils
python organize_test.py
```

Check for corrupted images:

```bash
python detect_bad_files.py
```

### 2. Training

Train the model:

```bash
cd src
python train.py
```

Training parameters:

- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 10
- Image size: 64x64

### 3. Testing

Evaluate model performance:

```bash
python test.py
```

This outputs the accuracy percentage on the test set.

### 4. Real-time Prediction

Run webcam prediction:

```bash
python predict_webcam.py
```

**Controls:**

- Place your hand in the blue rectangle (100,100 to 300,300)
- Press 'q' to quit
- Close the window to exit

## Model Performance

The model saves checkpoints after each epoch as `models/model_epoch_X.pth`. The final model (`model_epoch_10.pth`) is used for inference.

## Data Preprocessing

Images are preprocessed with:

- Resize to 64x64 pixels
- Convert to tensor
- Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for training
- **CPU**: Falls back to CPU if CUDA unavailable
- **Camera**: Webcam for real-time prediction
