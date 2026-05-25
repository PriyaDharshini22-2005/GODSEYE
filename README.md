# GODSEYE Framework

## Overview

GODSEYE is a comprehensive AI-powered framework developed for automated visual inspection of vehicle crankshaft components in industrial manufacturing environments. The framework provides a complete pipeline for dataset preparation, model training, configurable inference, real-time inspection, and multi-architecture experimentation.

The primary objective of GODSEYE is to improve defect detection accuracy while reducing the limitations of manual inspection processes. The framework combines YOLO-based object detection with multiple custom backbone-neck-head architectures to improve robustness across different surface textures, lighting conditions, and defect variations.

GODSEYE is designed to support practical production workflows by enabling:
- Real-time defect inspection
- Live camera-based detection
- Image-based inference
- Multi-model benchmarking
- Feature-map analysis
- Architecture-level experimentation
- Performance logging and comparison

---

# GitHub About

AI-powered crankshaft defect inspection framework featuring five custom detection architectures, unified training and inference workflows, live and file-based inspection modes, and architecture-level benchmarking for industrial defect detection research.

---

# Problem Statement

In industrial manufacturing, crankshaft inspection is a critical quality-control process. Manual inspection methods are difficult to scale because many defects are:
- Extremely small
- Texture-sensitive
- Lighting-dependent
- Difficult to detect consistently

Traditional visual inspection also suffers from:
- Human fatigue
- Inconsistent judgement
- Reduced inspection speed
- Limited scalability in high-volume production lines

Surface defects such as:
- Dent
- Peel-off
- Scratch
can significantly affect crankshaft reliability and mechanical performance.

GODSEYE addresses these challenges using deep-learning-based computer vision techniques capable of learning subtle visual defect patterns across varying production conditions.

---

# Framework Objectives

The framework is developed with the following objectives:

- Automate crankshaft defect inspection
- Improve inspection consistency
- Reduce false positives and false negatives
- Support architecture experimentation
- Enable fast industrial inference
- Provide explainable feature-level analysis
- Compare multiple deep-learning designs under a unified system

---

# Core Features

## Unified CLI Workflow
GODSEYE provides a single command-line interface for:
- Dataset preparation
- Model training
- Model inference
- Architecture execution
- Result analysis

---

## Automated Dataset Handling
The framework supports:
- Dataset path configuration
- Class metadata management
- Automatic dataset splitting

### Dataset Split Ratio
- Training Set: 70%
- Validation Set: 20%
- Test Set: 10%

---

## Data Augmentation Pipeline
Optional augmentation techniques can be applied during training to improve model generalization.

Supported augmentations include:
- Blur
- Rotation
- Horizontal Flip
- Vertical Flip

---

## YOLO Integration
The framework supports multiple YOLO versions:
- YOLOv9
- YOLOv10
- YOLOv11

Users can select the required YOLO version during execution.

---

## Flexible Architecture Execution
Users can:
- Run the default YOLO pipeline
- Execute a single architecture
- Execute all architectures simultaneously

This helps compare model performance across different defect scenarios.

---

## Advanced Logging System
The framework automatically generates:
- Architecture-specific logs
- Detection outputs
- Feature-map traces
- Statistical summaries
- Visual prediction artifacts

These outputs help during debugging, benchmarking, and performance analysis.

---

# Workflow

# 1. Training Workflow

The training pipeline allows users to prepare datasets, configure models, and train custom architectures.

## Training Steps

### Step 1: Dataset Configuration
Provide:
- Dataset path
- Class names
- Label configuration

---

### Step 2: Model Selection
Choose:
- Default YOLO model
- Custom pretrained weights
- One of the five custom architectures

---

### Step 3: Augmentation Selection
Optionally enable:
- Blur augmentation
- Rotation augmentation
- Flip augmentation

---

### Step 4: Hyperparameter Tuning
Users can configure:
- Epochs
- Learning rate
- Patience
- Batch size
- Other training parameters

---

### Step 5: Automatic YAML Generation
The framework automatically creates:
```yaml
data.yaml
```

This file contains:
- Dataset paths
- Class metadata
- Training configuration

---

### Step 6: Training Execution
After configuration:
- Training scripts are triggered automatically
- Logs and checkpoints are stored
- Outputs are organized per architecture

---

# 2. Detection Workflow

The inference pipeline supports both offline and real-time inspection workflows.

---

## Input Sources

### Live Camera Inspection
Supports real-time visual inspection using:
- Webcam feed
- Industrial camera feed

---

### File-Based Inspection
Supports inference on:
- Single images
- Captured inspection frames

---

## Execution Modes

### Default YOLO Mode
Runs standard YOLO inference pipeline.

---

### Single Architecture Mode
Executes one selected custom architecture.

---

### Multi-Architecture Mode
Runs all five architectures simultaneously for:
- Benchmarking
- Comparative analysis
- Ensemble-style evaluation

---

# Architecture Designs

GODSEYE contains five custom architecture implementations. Each architecture follows a unique design philosophy to improve feature diversity and defect recognition capability.

---

# Architecture 1

## Configuration
- Backbone: ResNet50
- Neck: FPN-based feature fusion
- Head: Custom YOLO detection head

## Design Philosophy
This architecture focuses on structured multi-stage feature extraction.

## Characteristics
- Hierarchical feature learning
- Strong feature-map preservation
- Detailed feature export pipeline
- Enhanced defect localization

---

# Architecture 2

## Configuration
- Backbone: ResNet50
- Neck: BiFPN-inspired aggregation
- Head: SiLU-activated YOLO head

## Design Philosophy
This model emphasizes efficient multi-scale feature aggregation.

## Characteristics
- Improved feature propagation
- Better scale-aware detection
- Layer-wise statistical logging
- Strong small-defect sensitivity

---

# Architecture 3

## Configuration
- Backbone: MobileNetV2
- Neck: SPP + CSP-inspired fusion
- Head: Mish-activated YOLO head

## Design Philosophy
This architecture prioritizes lightweight computation while preserving contextual understanding.

## Characteristics
- Lightweight backbone
- Fast inference speed
- Rich spatial pyramid pooling
- Improved contextual representation

---

# Architecture 4

## Configuration
- Backbone: EfficientNet-B0
- Neck: Enhanced FPN + PANet hybrid
- Head: Mish-based YOLO head

## Design Philosophy
This architecture balances efficiency and detection accuracy.

## Characteristics
- Deep feature refinement
- Efficient feature reuse
- Balanced computational cost
- Strong generalization capability

---

# Architecture 5

## Configuration
- Backbone: Custom Darknet-style stack
- Neck: ASPP multi-dilation context module
- Head: Probabilistic YOLO detection head

## Design Philosophy
This architecture focuses on contextual feature understanding and regularized learning.

## Characteristics
- Multi-dilation receptive fields
- Context-heavy representation learning
- Dropout regularization
- Improved difficult-pattern detection

---

# Why GODSEYE Improves Accuracy

## Multi-Architecture Strategy
Different architectures respond differently to:
- Texture variation
- Lighting conditions
- Defect scales
- Surface complexity

Using multiple architectures increases overall detection robustness.

---

## Diverse Feature Representation
Each architecture extracts features differently through:
- Different backbones
- Different fusion strategies
- Different activation mechanisms
- Different receptive fields

This improves defect coverage across multiple visual conditions.

---

## Flexible Model Benchmarking
Users can compare architectures and select:
- Highest accuracy model
- Fastest inference model
- Most stable model
- Best model for specific production environments

---

## Logging-Driven Optimization
The framework stores:
- Feature statistics
- Prediction traces
- Visual outputs
- Detection logs

This helps identify:
- False positives
- False negatives
- Weak feature regions
- Architecture limitations

---

# Project Structure

```bash
GODSEYE/
│
├── app.py
├── app.bat
├── requirements.txt
├── .config
│
├── inference/
│   ├── architecture_1.py
│   ├── architecture_2.py
│   ├── architecture_3.py
│   ├── architecture_4.py
│   ├── architecture_5.py
│   ├── live_inference.py
│   └── yolo_runner.py
│
├── weights/
│   ├── checkpoints
│   └── trained_models
│
├── logsfolder/
│   ├── architecture_logs
│   ├── feature_maps
│   ├── detection_outputs
│   └── statistical_reports
│
└── GodsEye_Final/
    ├── datasets
    ├── labels
    └── experimental_assets
```

---

# Quick Start

# Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Step 2: Configure Environment

Update the `.config` file with:
- Dataset path
- Input image path
- Model weight paths
- Camera configuration (optional)

---

# Step 3: Launch Framework

```bash
python app.py
```

---

# Step 4: Select Workflow

Choose:
- Training Mode
- Detection Mode

Follow the interactive prompts displayed in the CLI.

---

# Intended Applications

GODSEYE is designed for:
- Crankshaft quality inspection
- Industrial defect detection
- Manufacturing quality control
- Surface anomaly detection
- AI-assisted visual inspection systems

The framework can also be adapted for:
- Metal surface inspection
- Mechanical component inspection
- Industrial automation pipelines
- Other machine-vision defect-detection tasks

---

# Experimental and Research Usage

The repository contains multiple architecture implementations intended for:
- Research experimentation
- Industrial benchmarking
- Architecture comparison
- Feature-analysis studies
- Detection optimization

---

# Production Deployment Note

Before deploying in a production environment, validate:
- Hardware throughput
- GPU utilization
- Inference latency
- Real-time processing speed
- Class-wise precision and recall
- Long-term inspection consistency

Performance may vary depending on:
- Camera quality
- Lighting conditions
- Surface texture
- Production-line speed
- Dataset quality

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- YOLOv9
- YOLOv10
- YOLOv11
- Deep Learning
- Computer Vision
- CNN Architectures

---

# Future Enhancements

Planned improvements include:
- TensorRT optimization
- Edge-device deployment
- ONNX export support
- Real-time dashboard integration
- Cloud-based monitoring
- Automated report generation
- Defect severity estimation
- Industrial IoT integration
- Multi-camera inspection support

---

# License

This project is intended for:
- Research
- Educational purposes
- Industrial AI experimentation
- Computer vision benchmarking

Please validate the framework thoroughly before production deployment.
