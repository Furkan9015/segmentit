#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm

from segmentit.data.fast5_reader import Fast5Reader
from segmentit.data.label_reader import LabelReader
from segmentit.utils.signal_processing import normalize_signal, median_filter_signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prepare nanopore signal data and save to HDF5')
    parser.add_argument('--fast5_dir', type=str, required=True, help='Directory containing Fast5 files')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to labels TSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output HDF5 file')
    parser.add_argument('--chunk_size', type=int, default=4000, help='Size of signal chunks')
    parser.add_argument('--chunk_stride', type=int, default=2000, help='Stride between signal chunks')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of worker threads')
    parser.add_argument('--min_segments', type=int, default=1, help='Minimum segments in chunk to include')
    parser.add_argument('--compression', type=str, default='gzip', help='HDF5 compression type (gzip, lzf, or none)')
    return parser.parse_args()

def prepare_signal_chunk(signal_data, chunk_idx, chunk_size, chunk_stride, labels=None):
    """Process a single chunk of signal data."""
    start_idx = chunk_idx * chunk_stride
    end_idx = start_idx + chunk_size
    
    if end_idx > len(signal_data):
        # Pad with zeros if needed
        chunk = np.zeros(chunk_size)
        chunk[:len(signal_data) - start_idx] = signal_data[start_idx:]
    else:
        chunk = signal_data[start_idx:end_idx]
    
    # Normalize and filter
    chunk = normalize_signal(chunk)
    chunk = median_filter_signal(chunk)
    
    # Create corresponding label if available
    if labels is not None:
        label_chunk = np.zeros(chunk_size)
        for start, end in labels:
            if end >= start_idx and start < end_idx:
                # Adjust indices to chunk coordinates
                s = max(0, start - start_idx)
                e = min(chunk_size, end - start_idx)
                # Mark segment boundaries
                if s < chunk_size:
                    label_chunk[s] = 1
                if e < chunk_size:
                    label_chunk[e] = 1
        return chunk, label_chunk
    
    return chunk

def process_read(read_id, signal, labels, chunk_size, chunk_stride, min_segments):
    """Process a complete read and its chunks."""
    n_chunks = max(1, (len(signal) - chunk_size) // chunk_stride + 1)
    
    valid_chunks = []
    valid_labels = []
    chunk_indices = []
    
    for i in range(n_chunks):
        start_idx = i * chunk_stride
        end_idx = start_idx + chunk_size
        
        # Count segments in this chunk
        segment_count = sum(1 for start, end in labels 
                           if (start >= start_idx and start < end_idx) or 
                              (end >= start_idx and end < end_idx))
        
        if segment_count >= min_segments:
            chunk, label = prepare_signal_chunk(signal, i, chunk_size, chunk_stride, labels)
            valid_chunks.append(chunk)
            valid_labels.append(label)
            chunk_indices.append(i)
    
    if valid_chunks:
        return read_id, np.stack(valid_chunks), np.stack(valid_labels), np.array(chunk_indices)
    return None

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize readers
    logger.info("Initializing Fast5 and Label readers...")
    fast5_reader = Fast5Reader(args.fast5_dir)
    label_reader = LabelReader(args.labels_path)
    
    # Get all read IDs that have both signal and labels
    read_ids = set(fast5_reader.read_id_map.keys()).intersection(label_reader.read_id_map.keys())
    logger.info(f"Found {len(read_ids)} reads with both signal and labels")
    
    # Create HDF5 file
    compression = args.compression if args.compression.lower() != 'none' else None
    with h5py.File(output_path, 'w') as f:
        # Create dataset groups
        signals_group = f.create_group('signals')
        labels_group = f.create_group('labels')
        metadata_group = f.create_group('metadata')
        
        # Store dataset parameters
        params = metadata_group.create_group('parameters')
        params.attrs['chunk_size'] = args.chunk_size
        params.attrs['chunk_stride'] = args.chunk_stride
        params.attrs['min_segments'] = args.min_segments
        
        # Store read_ids
        metadata_group.create_dataset('read_ids', data=np.array(list(read_ids), dtype='S36'))
        
        # Process reads in parallel
        total_chunks = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            
            for read_id in read_ids:
                signal = fast5_reader.get_signal(read_id)
                labels = label_reader.get_labels(read_id)
                
                futures.append(
                    executor.submit(
                        process_read, 
                        read_id, 
                        signal, 
                        labels, 
                        args.chunk_size, 
                        args.chunk_stride,
                        args.min_segments
                    )
                )
            
            # Process results as they complete
            for future in tqdm(futures, desc="Processing reads"):
                result = future.result()
                if result is None:
                    continue
                    
                read_id, chunks, chunk_labels, chunk_indices = result
                
                # Store in HDF5
                read_group = signals_group.create_group(read_id)
                read_group.create_dataset('chunks', data=chunks, compression=compression)
                read_group.create_dataset('indices', data=chunk_indices, compression=compression)
                
                label_group = labels_group.create_group(read_id)
                label_group.create_dataset('labels', data=chunk_labels, compression=compression)
                
                total_chunks += len(chunks)
        
        logger.info(f"Processed {total_chunks} valid chunks from {len(read_ids)} reads")
        logger.info(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()