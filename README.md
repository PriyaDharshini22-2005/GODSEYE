# GODSEYE Framework

GODSEYE is an AI framework for automated visual inspection of vehicle crankshaft parts. It supports end-to-end model training, configurable inference, live camera inspection, and architecture-level experimentation to improve defect detection accuracy in real industrial workflows.

The framework is designed around practical production needs:
- Detect surface defects such as dent, peel-off, and scratch.
- Run fast inspection from image files or live camera feeds.
- Compare multiple deep-learning architecture styles under one unified workflow.
- Generate architecture-specific logs, feature-map traces, and visual output artifacts for analysis.

## GitHub About (Short Description)
AI framework for crankshaft defect inspection with 5 custom detection architectures, training plus live/file inference modes, and accuracy-focused multi-architecture experimentation.

## Problem Scope
Crankshaft quality inspection is difficult to scale manually because defects are small, varied, and sensitive to lighting and surface texture. GODSEYE addresses this by combining YOLO-based detection with custom backbone-neck-head variants, enabling more reliable detection across different defect patterns.

## Core Features
- Unified CLI flow for training and detection in a single entry point.
- Dataset handling with class metadata input and automatic train/val/test split (70/20/10).
- Optional augmentation pipeline (blur, rotate, flips).
- Default YOLO execution path with selectable YOLOv9/YOLOv10/YOLOv11.
- Architecture runner for one selected architecture or all architectures.
- Side-by-side result display and per-architecture output logging.

## Workflow Options
### 1) Training Mode
- Provide dataset path and class details.
- Choose custom model path or one of five architecture slots.
- Optionally apply augmentation.
- Optionally fine-tune training parameters (epochs, learning rate, patience, and others).
- Auto-generate data.yaml and trigger training script.

### 2) Detection Mode
- Choose input source:
- Live camera capture.
- Single file path.
- Choose execution mode:
- Default YOLO mode.
- Run one specific architecture.
- Run all five architectures.

## Five Architecture Styles
Each architecture is built with a different design philosophy, so the framework can capture complementary strengths and improve practical detection accuracy over a single fixed model setup.

### Architecture 1 
- Backbone: ResNet50 staged feature extraction.
- Neck: FPN-style fusion.
- Head: Custom YOLO head.
- Style: structured multi-stage feature flow with detailed feature-map export.

### Architecture 2 
- Backbone: ResNet50.
- Neck: BiFPN-like feature aggregation.
- Head: SiLU-based YOLO head.
- Style: strong multi-scale fusion with layer-wise statistical logging.

### Architecture 3 
- Backbone: MobileNetV2.
- Neck: SPP plus CSP-inspired fusion.
- Head: Mish-activated detection head.
- Style: lightweight backbone with rich pyramid pooling for robust context capture.

### Architecture 4 
- Backbone: EfficientNet-B0.
- Neck: Enhanced FPN plus PANet hybrid.
- Head: Mish-based YOLO head.
- Style: efficiency-accuracy balance with deep neck refinement.

### Architecture 5 
- Backbone: Darknet-style custom stack with dropout regularization.
- Neck: ASPP (multi-dilation context).
- Head: YOLO head with probabilistic output mapping.
- Style: context-heavy and regularized representation for difficult defect patterns.

## Why Accuracy Improves
- Multi-architecture strategy: different backbone/neck/head combinations respond differently to subtle defect textures.
- Better feature diversity: each pipeline emphasizes different scales and receptive fields.
- Practical model selection: users can run one or all architectures and choose the best performer for their target line conditions.
- Logging-first design: feature statistics and outputs make it easier to diagnose false positives/false negatives and iteratively improve results.

## Project Structure
- app.py: Main framework entry for training and detection options.
- app.bat: Training launcher helper.
- inference/: Architecture scripts, live inference, and default YOLO runner.
- weights/: Model and checkpoint files.
- logsfolder/: Architecture-specific logs and generated outputs.
- GodsEye_Final/: Label and asset collection used in experiments.

## Quick Start
1. Install dependencies from requirements.txt.
2. Update .config with paths for dataset, input image, and model weights.
3. Run app.py.
4. Select Training or Detection and follow the interactive prompts.

## Intended Use
This framework is intended for AI-assisted visual quality inspection of crankshaft and related machined vehicle parts. It can be adapted to other industrial defect-detection domains with compatible datasets and labels.

## Note
The repository contains multiple architecture implementations and logs intended for experimentation and benchmarking. For production deployment, validate hardware throughput, latency targets, and class-wise quality metrics in your own plant environment.
