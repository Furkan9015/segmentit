#!/usr/bin/env python
"""Inference script for nanopore segmentation models."""

import argparse
import yaml
import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentit.data.fast5_reader import Fast5Reader
from segmentit.models.inference import SegmentationInference
from segmentit.utils.metrics import visualize_signal_with_boundaries
from segmentit.utils.signal_processing import normalize_signal, median_filter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger('inference')

def run_inference(args):
    """Run inference with segmentation model.
    
    Args:
        args: Command line arguments
    """
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    logger.info("Loading data...")
    fast5_dir = Path(args.fast5_dir)
    
    # Initialize Fast5Reader
    fast5_reader = Fast5Reader(fast5_dir)
    read_ids = fast5_reader.get_all_read_ids()
    
    if args.read_id:
        # Process specific read ID
        if args.read_id not in read_ids:
            logger.error(f"Read ID {args.read_id} not found")
            return
        read_ids = [args.read_id]
    elif args.num_reads > 0:
        # Limit number of reads
        read_ids = list(read_ids)[:args.num_reads]
    
    logger.info(f"Found {len(read_ids)} reads")
    
    # Initialize inference
    inference = SegmentationInference(
        model_path=args.model_path,
        device=device,
        use_onnx=args.use_onnx,
        chunk_size=args.chunk_size,
        stride=args.stride,
        normalize=True,
        filter_signal=True,
        threshold=args.threshold,
    )
    
    # Process reads
    all_results = {}
    
    if args.parallel:
        # Process reads in parallel
        logger.info(f"Processing {len(read_ids)} reads in parallel...")
        results = inference.process_multiple_reads(
            fast5_reader, read_ids, max_workers=args.num_workers
        )
        all_results.update(results)
    else:
        # Process reads sequentially
        logger.info(f"Processing {len(read_ids)} reads sequentially...")
        for read_id in tqdm(read_ids):
            result = inference.process_read(fast5_reader, read_id)
            if result is not None:
                all_results[read_id] = result
    
    # Save results
    results_path = output_dir / "inference_results.json"
    
    # Convert results to serializable format
    serializable_results = {}
    for read_id, result in all_results.items():
        serializable_results[read_id] = {
            'read_id': read_id,
            'signal_length': int(result['signal_length']),
            'boundary_positions': result['boundary_positions'].tolist(),
            'num_boundaries': len(result['boundary_positions']),
        }
    
    with open(results_path, 'w') as f:
        json.dump({
            'results': serializable_results,
            'inference_settings': {
                'threshold': args.threshold,
                'num_reads': len(read_ids),
                'chunk_size': args.chunk_size,
                'stride': args.stride,
            }
        }, f, indent=2)
    
    logger.info(f"Saved inference results to {results_path}")
    
    # Visualize examples if requested
    if args.visualize:
        logger.info("Visualizing examples...")
        for i, (read_id, result) in enumerate(all_results.items()):
            if i >= args.num_visualize:
                break
                
            # Get signal
            signal = fast5_reader.get_signal_by_read_id(read_id)
            if signal is None:
                continue
            
            # Get binary boundaries
            binary_boundaries = result['binary_boundaries']
            
            # Visualize segment boundaries
            vis_path = output_dir / f"visualization_{read_id}.png"
            
            # For very long signals, visualize a subset
            if len(signal) > 10000:
                # Find a region with boundaries
                regions = []
                for i in range(0, len(signal), 5000):
                    end = min(i + 5000, len(signal))
                    if np.sum(binary_boundaries[i:end]) > 0:
                        regions.append((i, end))
                        if len(regions) >= 3:  # Limit to 3 regions
                            break
                
                # Visualize each region
                for idx, (start, end) in enumerate(regions):
                    vis_path = output_dir / f"visualization_{read_id}_region{idx}.png"
                    visualize_signal_with_boundaries(
                        signal[start:end],
                        np.zeros_like(binary_boundaries[start:end]),  # No ground truth
                        binary_boundaries[start:end],
                        output_path=vis_path
                    )
            else:
                # Visualize the whole signal
                visualize_signal_with_boundaries(
                    signal,
                    np.zeros_like(binary_boundaries),  # No ground truth
                    binary_boundaries,
                    output_path=vis_path
                )
    
    # Benchmark inference speed
    if args.benchmark:
        logger.info("Benchmarking inference speed...")
        inference.benchmark_speed(signal_length=400000, num_runs=5)
    
    # Export segmentation to TSV if requested
    if args.export_tsv:
        export_path = output_dir / "segmentation.tsv"
        logger.info(f"Exporting segmentation to {export_path}...")
        
        with open(export_path, 'w') as f:
            # Write header
            f.write("read_id\tkmer_idx\tstart_raw_idx\tend_raw_idx\n")
            
            # Write segments for each read
            for read_id, result in all_results.items():
                boundary_positions = result['boundary_positions']
                
                # Convert boundary positions to segments
                if len(boundary_positions) >= 2:
                    for i in range(len(boundary_positions) - 1):
                        start_idx = int(boundary_positions[i])
                        end_idx = int(boundary_positions[i + 1])
                        f.write(f"{read_id}\t{i}\t{start_idx}\t{end_idx}\n")
        
        logger.info(f"Exported segmentation to {export_path}")

def process_file(args):
    """Process a single Fast5 file without using Fast5Reader.
    
    This is for demonstration of real-time processing.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize inference
    inference = SegmentationInference(
        model_path=args.model_path,
        device=device,
        use_onnx=args.use_onnx,
        chunk_size=args.chunk_size,
        stride=args.stride,
        normalize=True,
        filter_signal=True,
        threshold=args.threshold,
    )
    
    # Open Fast5 file
    fast5_path = Path(args.file)
    if not fast5_path.exists():
        logger.error(f"File not found: {fast5_path}")
        return
    
    try:
        with h5py.File(fast5_path, 'r') as f5:
            # Process each read in the file
            for read_group in f5:
                # Extract read_id
                try:
                    read_id = f5[read_group].attrs.get('read_id')
                    if read_id is None:
                        # Try alternative location
                        read_id = f5[read_group].attrs.get('read_id')
                    if isinstance(read_id, bytes):
                        read_id = read_id.decode('utf-8')
                        
                    if not read_id:
                        logger.warning(f"Read ID not found for {read_group}")
                        continue
                    
                    # Extract signal
                    try:
                        # ONT format
                        signal = f5[f"{read_group}/Raw/Signal"][:]
                    except KeyError:
                        # Try alternative path
                        try:
                            signal = f5[f"{read_group}/Raw"][:]
                        except KeyError:
                            logger.warning(f"Signal not found for {read_id}")
                            continue
                    
                    # Process signal
                    logger.info(f"Processing read {read_id} with {len(signal)} datapoints...")
                    start_time = time.time()
                    result = inference.process_signal(signal, read_id)
                    end_time = time.time()
                    
                    # Log results
                    processing_time = end_time - start_time
                    boundaries = result['boundary_positions']
                    logger.info(f"Found {len(boundaries)} boundaries in {processing_time:.2f} seconds")
                    logger.info(f"Processing speed: {len(signal)/processing_time:.2f} datapoints/second")
                    
                    # Write results to file
                    output_path = output_dir / f"{read_id}_segments.json"
                    with open(output_path, 'w') as f:
                        json.dump({
                            'read_id': read_id,
                            'signal_length': len(signal),
                            'boundaries': boundaries.tolist(),
                            'processing_time': processing_time,
                            'processing_speed': len(signal)/processing_time,
                        }, f, indent=2)
                    
                    # Visualize if requested
                    if args.visualize:
                        vis_path = output_dir / f"{read_id}_visualization.png"
                        visualize_signal_with_boundaries(
                            signal,
                            np.zeros_like(result['binary_boundaries']),  # No ground truth
                            result['binary_boundaries'],
                            output_path=vis_path
                        )
                    
                except Exception as e:
                    logger.error(f"Error processing read {read_group}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error opening file {fast5_path}: {e}")

def simulate_live_processing(args):
    """Simulate live processing of a stream of signal data.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize inference
    inference = SegmentationInference(
        model_path=args.model_path,
        device=device,
        use_onnx=args.use_onnx,
        chunk_size=args.chunk_size,
        stride=args.stride,
        normalize=True,
        filter_signal=True,
        threshold=args.threshold,
    )
    
    # Create synthetic signal
    np.random.seed(42)
    signal_length = 1000000  # 1M datapoints
    
    # Create synthetic signal with segment boundaries
    segment_length_mean = 50
    segment_length_std = 10
    
    # Generate segments
    segments = []
    current_pos = 0
    
    while current_pos < signal_length:
        segment_length = max(10, int(np.random.normal(segment_length_mean, segment_length_std)))
        segments.append((current_pos, current_pos + segment_length))
        current_pos += segment_length
    
    # Generate signal
    signal = np.random.normal(0, 1, signal_length)
    
    # Add level shifts at segment boundaries
    current_level = 0
    for start, end in segments:
        current_level += np.random.normal(0, 0.5)
        signal[start:end] += current_level
    
    # Add noise
    signal += np.random.normal(0, 0.2, signal_length)
    
    # Create ground truth boundaries
    true_boundaries = np.zeros(signal_length, dtype=np.int32)
    for start, _ in segments:
        if start < signal_length:
            true_boundaries[start] = 1
    
    logger.info(f"Created synthetic signal with {len(segments)} segments")
    
    # Process signal in streaming manner
    chunk_size = args.chunk_size * 10  # Larger chunks for simulation
    buffer = []
    all_predictions = []
    all_boundaries = []
    
    logger.info(f"Processing signal in streaming manner with chunk size {chunk_size}...")
    for i in range(0, signal_length, chunk_size):
        # Get next chunk
        end = min(i + chunk_size, signal_length)
        current_chunk = signal[i:end]
        
        # Add to buffer
        buffer.extend(current_chunk)
        
        # Process if buffer is large enough
        if len(buffer) >= args.chunk_size * 2:
            logger.info(f"Processing buffer with {len(buffer)} datapoints...")
            
            # Process buffer
            result = inference.process_signal(np.array(buffer))
            
            # Get predictions
            predictions = result['binary_boundaries']
            boundaries = result['boundary_positions']
            
            # Store predictions (adjusted for global position)
            all_predictions.extend(predictions)
            all_boundaries.extend(boundaries + (i - len(buffer)))
            
            # Keep the last chunk in the buffer for overlap
            buffer = buffer[-args.chunk_size:]
    
    # Process remaining buffer
    if buffer:
        result = inference.process_signal(np.array(buffer))
        predictions = result['binary_boundaries']
        boundaries = result['boundary_positions']
        
        all_predictions.extend(predictions)
        all_boundaries.extend(boundaries + (signal_length - len(buffer)))
    
    # Compare with ground truth
    all_predictions = np.array(all_predictions[:signal_length])
    
    # Calculate metrics
    from segmentit.utils.metrics import compute_tolerance_metrics
    precision, recall, f1 = compute_tolerance_metrics(
        true_boundaries, all_predictions, tolerance=5
    )
    
    logger.info(f"Streaming processing results:")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Visualize a segment
    vis_range = 5000
    visualize_signal_with_boundaries(
        signal[:vis_range],
        true_boundaries[:vis_range],
        all_predictions[:vis_range],
        output_path=Path(args.output_dir) / "streaming_visualization.png"
    )

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run inference with nanopore segmentation model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    parser.add_argument('--output_dir', type=str, default='inference_output', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--use_onnx', action='store_true', help='Use ONNX runtime for inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for boundary detection')
    parser.add_argument('--chunk_size', type=int, default=4000, help='Chunk size for inference')
    parser.add_argument('--stride', type=int, default=2000, help='Stride for inference')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--num_visualize', type=int, default=5, help='Number of reads to visualize')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark inference speed')
    parser.add_argument('--export_tsv', action='store_true', help='Export segmentation to TSV')
    
    # Options for batch processing
    parser.add_argument('--fast5_dir', type=str, help='Path to fast5 directory')
    parser.add_argument('--read_id', type=str, help='Process specific read ID')
    parser.add_argument('--num_reads', type=int, default=0, help='Number of reads to process (0 for all)')
    parser.add_argument('--parallel', action='store_true', help='Process reads in parallel')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for parallel processing')
    
    # Options for single file processing
    parser.add_argument('--file', type=str, help='Process a single Fast5 file')
    
    # Option for streaming simulation
    parser.add_argument('--simulate_streaming', action='store_true', help='Simulate streaming processing')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check arguments
    if args.simulate_streaming:
        simulate_live_processing(args)
    elif args.file:
        process_file(args)
    elif args.fast5_dir:
        run_inference(args)
    else:
        parser.error("Either --fast5_dir, --file, or --simulate_streaming must be provided")

if __name__ == "__main__":
    main()