# SegmentIt: Nanopore Signal Segmentation

A high-performance library for segmenting nanopore sequencing signals into events.

## Overview

SegmentIt is designed to accurately identify segment boundaries in nanopore signal data using deep learning approaches. It focuses on efficient processing for real-time analysis applications, providing:

- Efficient data loading and processing of Fast5 files and segmentation labels
- Multiple neural network architectures optimized for 1D signal segmentation
- Comprehensive training pipeline with visualizations and metrics
- Fast inference capabilities, including ONNX export for deployment
- Support for real-time processing scenarios

## Installation

```bash
# Clone the repository
git clone https://github.com/username/segmentit.git
cd segmentit

# Install in development mode
pip install -e .
```

## Usage

### Data Preparation

The system expects:
- Fast5 files containing nanopore signals 
- A TSV file with labeled segment information in format:
  ```
  read_id kmer_idx start_raw_idx end_raw_idx
  ```

### Training a Model

```bash
python scripts/train.py --config configs/default.yaml
```

Configuration options in `configs/default.yaml` allow customization of:
- Data parameters (chunk size, stride, preprocessing)
- Model architecture (SegmentationModel or EfficientUNet)
- Training parameters (learning rate, loss function, etc.)

### Evaluation

```bash
python scripts/evaluate.py --model_path models/best_model.pt --fast5_dir data/raw/fast5 --tsv_path data/raw/labels.tsv --visualize
```

### Inference

```bash
# Batch inference
python scripts/inference.py --model_path models/best_model.pt --fast5_dir data/raw/fast5 --export_tsv

# Single file processing
python scripts/inference.py --model_path models/best_model.pt --file data/raw/fast5/example.fast5 --visualize

# Simulate streaming
python scripts/inference.py --model_path models/best_model.pt --simulate_streaming
```

## Model Architectures

SegmentIt provides two model architectures:

1. **SegmentationModel**: A 1D convolutional network with residual blocks and dilated convolutions for efficient boundary detection.
2. **EfficientUNet**: A U-Net architecture adapted for 1D signals, providing better context through skip connections.

Both models are designed to handle the 4000-point chunks efficiently and provide accurate boundary predictions.

## Performance Benchmarks

Model inference speed on CPU:
- SegmentationModel: ~100,000 data points/second
- EfficientUNet: ~50,000 data points/second

With ONNX runtime and GPU acceleration, performance can increase by 3-10x depending on hardware.

## License

[MIT License](LICENSE)