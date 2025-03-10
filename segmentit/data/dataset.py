"""Dataset classes for nanopore segmentation."""

from typing import Dict, List, Tuple, Optional, Set, Iterator, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from segmentit.data.fast5_reader import Fast5Reader
from segmentit.data.label_reader import LabelReader
from segmentit.utils.signal_processing import normalize_signal, median_filter

logger = logging.getLogger(__name__)

class NanoporeSegmentationDataset(Dataset):
    """Dataset for nanopore signal segmentation with optimized performance."""
    
    def __init__(
        self,
        fast5_reader: Fast5Reader,
        label_reader: LabelReader,
        chunk_size: int = 4000,
        stride: int = 2000,
        max_workers: int = 4,
        normalize: bool = True,
        filter_signal: bool = True,
        augment: bool = False,
        cache_size: int = 1000,
    ):
        """Initialize the dataset.
        
        Args:
            fast5_reader: Fast5Reader instance for signal access
            label_reader: LabelReader instance for segment labels
            chunk_size: Size of chunks to extract from signals
            stride: Stride for sliding window when creating chunks
            max_workers: Max number of workers for parallel processing
            normalize: Whether to normalize signal chunks
            filter_signal: Whether to apply median filtering to signal
            augment: Whether to apply data augmentation
            cache_size: Size of in-memory cache for chunks
        """
        self.fast5_reader = fast5_reader
        self.label_reader = label_reader
        self.chunk_size = chunk_size
        self.stride = stride
        self.max_workers = max_workers
        self.normalize = normalize
        self.filter_signal = filter_signal
        self.augment = augment
        self.cache_size = cache_size
        
        # Find common read IDs between signal and labels
        self.read_ids = sorted(list(
            set(self.fast5_reader.get_all_read_ids()) & 
            self.label_reader.get_read_ids()
        ))
        
        logger.info(f"Found {len(self.read_ids)} reads with both signal and labels")
        
        # Prepare chunks for all read_ids
        self.chunks: List[Tuple[str, int, int]] = []  # (read_id, start_idx, end_idx)
        self._prepare_chunks()
        
        # Setup LRU cache for commonly accessed chunks
        self._signal_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
        self._label_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
        
    def _prepare_chunks(self) -> None:
        """Prepare chunk indices for all reads."""
        logger.info("Preparing chunks for all reads...")
        
        # Function to process a single read
        def process_read(read_id: str) -> List[Tuple[str, int, int]]:
            signal = self.fast5_reader.get_signal_by_read_id(read_id)
            if signal is None:
                return []
                
            read_chunks = []
            signal_len = len(signal)
            
            # Use sliding window to create chunks
            for start_idx in range(0, signal_len - self.chunk_size + 1, self.stride):
                end_idx = start_idx + self.chunk_size
                # Check if there are any segments in this chunk
                segments = self.label_reader.get_segments_in_range(read_id, start_idx, end_idx)
                if segments is not None and len(segments) > 0:
                    read_chunks.append((read_id, start_idx, end_idx))
                
            return read_chunks
                
        # Process reads in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_lists = list(executor.map(process_read, self.read_ids))
            
        # Flatten the list of chunks
        self.chunks = [chunk for chunk_list in chunk_lists for chunk in chunk_list]
        
        # Shuffle chunks for better training
        random.shuffle(self.chunks)
        
        logger.info(f"Prepared {len(self.chunks)} chunks from {len(self.read_ids)} reads")
    
    def __len__(self) -> int:
        """Get the number of chunks in the dataset."""
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a chunk by index.
        
        Args:
            idx: Index of the chunk
            
        Returns:
            Dict containing signal and label tensors
        """
        read_id, start_idx, end_idx = self.chunks[idx]
        
        # Check cache first for signal
        cache_key = (read_id, start_idx, end_idx)
        if cache_key in self._signal_cache:
            signal = self._signal_cache[cache_key]
        else:
            # Get raw signal and process it
            full_signal = self.fast5_reader.get_signal_by_read_id(read_id)
            if full_signal is None:
                # This should not happen since we checked during preparation
                raise ValueError(f"Signal not found for read_id {read_id}")
            
            signal = full_signal[start_idx:end_idx].copy()
            
            # Apply preprocessing
            if self.filter_signal:
                signal = median_filter(signal, window_size=5)
            if self.normalize:
                signal = normalize_signal(signal)
                
            # Cache signal if cache not full
            if len(self._signal_cache) < self.cache_size:
                self._signal_cache[cache_key] = signal
        
        # Check cache for labels
        if cache_key in self._label_cache:
            labels = self._label_cache[cache_key]
        else:
            # Get boundary labels
            labels = self.label_reader.generate_boundary_labels(read_id, start_idx, end_idx)
            
            # Cache labels if cache not full
            if len(self._label_cache) < self.cache_size:
                self._label_cache[cache_key] = labels
        
        # Apply data augmentation if enabled
        if self.augment:
            signal = self._augment_signal(signal)
        
        # Convert to tensors
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return {
            'signal': signal_tensor,
            'labels': labels_tensor,
            'read_id': read_id,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
    
    def _augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply data augmentation to signal.
        
        Args:
            signal: Signal to augment
            
        Returns:
            Augmented signal
        """
        # Random scaling
        if random.random() < 0.5:
            scale_factor = random.uniform(0.8, 1.2)
            signal = signal * scale_factor
        
        # Random noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, size=signal.shape)
            signal = signal + noise
        
        # Random shift
        if random.random() < 0.3:
            shift = random.uniform(-0.1, 0.1)
            signal = signal + shift
            
        return signal
    
    def get_dataloader(
        self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4
    ) -> DataLoader:
        """Get a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
        )