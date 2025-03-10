#!/usr/bin/env python
"""Evaluation script for nanopore segmentation models."""

import argparse
import yaml
import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentit.data.fast5_reader import Fast5Reader
from segmentit.data.label_reader import LabelReader
from segmentit.data.dataset import NanoporeSegmentationDataset
from segmentit.models.inference import SegmentationInference
from segmentit.utils.metrics import compute_metrics, compute_tolerance_metrics
from segmentit.utils.metrics import visualize_signal_with_boundaries
from segmentit.utils.metrics import plot_precision_recall_curve, plot_roc_curve

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger('evaluate')

def evaluate_model(args):
    """Evaluate segmentation model.
    
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
    tsv_path = Path(args.tsv_path)
    
    fast5_reader = Fast5Reader(fast5_dir)
    label_reader = LabelReader(tsv_path)
    
    # Get common read IDs
    read_ids = sorted(list(
        set(fast5_reader.get_all_read_ids()) & 
        label_reader.get_read_ids()
    ))
    
    if args.num_reads > 0:
        # Limit number of reads for evaluation
        read_ids = read_ids[:args.num_reads]
    
    logger.info(f"Found {len(read_ids)} reads with both signal and labels")
    
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
    
    # Evaluate on all reads
    results = []
    metrics_all = {
        'precision': [],
        'recall': [],
        'f1': [],
        'precision_tol': [],
        'recall_tol': [],
        'f1_tol': [],
    }
    
    logger.info(f"Evaluating model on {len(read_ids)} reads...")
    for read_id in tqdm(read_ids):
        # Get signal
        signal = fast5_reader.get_signal_by_read_id(read_id)
        if signal is None:
            logger.warning(f"Signal not found for read_id {read_id}")
            continue
        
        # Get ground truth boundaries
        true_boundaries = label_reader.generate_boundary_labels(
            read_id, 0, len(signal)
        )
        
        # Process signal
        pred_result = inference.process_signal(signal, read_id)
        pred_boundaries = pred_result['binary_boundaries']
        
        # Compute metrics
        metrics = {
            'read_id': read_id,
            'signal_length': len(signal),
            'true_boundaries': int(np.sum(true_boundaries)),
            'pred_boundaries': int(np.sum(pred_boundaries)),
        }
        
        # Compute standard metrics
        precision, recall, f1 = compute_tolerance_metrics(
            true_boundaries, pred_boundaries, tolerance=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Compute tolerance metrics
        precision_tol, recall_tol, f1_tol = compute_tolerance_metrics(
            true_boundaries, pred_boundaries, tolerance=args.tolerance
        )
        metrics['precision_tol'] = precision_tol
        metrics['recall_tol'] = recall_tol
        metrics['f1_tol'] = f1_tol
        
        # Save metrics
        results.append(metrics)
        
        # Update overall metrics
        metrics_all['precision'].append(precision)
        metrics_all['recall'].append(recall)
        metrics_all['f1'].append(f1)
        metrics_all['precision_tol'].append(precision_tol)
        metrics_all['recall_tol'].append(recall_tol)
        metrics_all['f1_tol'].append(f1_tol)
        
        # Visualize example if requested
        if args.visualize and len(results) <= args.num_visualize:
            # Visualize segment boundaries
            vis_path = output_dir / f"visualization_{read_id}.png"
            
            # For very long signals, visualize a subset
            if len(signal) > 10000:
                # Find a region with boundaries
                regions = []
                for i in range(0, len(signal), 5000):
                    end = min(i + 5000, len(signal))
                    region_true = true_boundaries[i:end]
                    region_pred = pred_boundaries[i:end]
                    if np.sum(region_true) > 0 or np.sum(region_pred) > 0:
                        regions.append((i, end))
                        if len(regions) >= 3:  # Limit to 3 regions
                            break
                
                # Visualize each region
                for idx, (start, end) in enumerate(regions):
                    vis_path = output_dir / f"visualization_{read_id}_region{idx}.png"
                    visualize_signal_with_boundaries(
                        signal[start:end],
                        true_boundaries[start:end],
                        pred_boundaries[start:end],
                        output_path=vis_path
                    )
            else:
                # Visualize the whole signal
                visualize_signal_with_boundaries(
                    signal,
                    true_boundaries,
                    pred_boundaries,
                    output_path=vis_path
                )
    
    # Compute average metrics
    avg_metrics = {
        'precision': np.mean(metrics_all['precision']),
        'recall': np.mean(metrics_all['recall']),
        'f1': np.mean(metrics_all['f1']),
        'precision_tol': np.mean(metrics_all['precision_tol']),
        'recall_tol': np.mean(metrics_all['recall_tol']),
        'f1_tol': np.mean(metrics_all['f1_tol']),
    }
    
    # Print results
    logger.info("Evaluation results:")
    logger.info(f"Exact metrics: Precision={avg_metrics['precision']:.4f}, " +
               f"Recall={avg_metrics['recall']:.4f}, F1={avg_metrics['f1']:.4f}")
    logger.info(f"Tolerance metrics: Precision={avg_metrics['precision_tol']:.4f}, " +
               f"Recall={avg_metrics['recall_tol']:.4f}, F1={avg_metrics['f1_tol']:.4f}")
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'individual_results': results,
            'average_metrics': avg_metrics,
            'evaluation_settings': {
                'threshold': args.threshold,
                'tolerance': args.tolerance,
                'num_reads': len(read_ids),
            }
        }, f, indent=2)
    
    logger.info(f"Saved evaluation results to {results_path}")
    
    # Benchmark inference speed
    if args.benchmark:
        logger.info("Benchmarking inference speed...")
        inference.benchmark_speed(signal_length=400000, num_runs=5)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate a nanopore segmentation model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    parser.add_argument('--fast5_dir', type=str, required=True, help='Path to fast5 directory')
    parser.add_argument('--tsv_path', type=str, required=True, help='Path to TSV file with labels')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--use_onnx', action='store_true', help='Use ONNX runtime for inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for boundary detection')
    parser.add_argument('--tolerance', type=int, default=5, help='Position tolerance for metrics')
    parser.add_argument('--chunk_size', type=int, default=4000, help='Chunk size for inference')
    parser.add_argument('--stride', type=int, default=2000, help='Stride for inference')
    parser.add_argument('--num_reads', type=int, default=0, help='Number of reads to evaluate (0 for all)')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--num_visualize', type=int, default=5, help='Number of reads to visualize')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark inference speed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()