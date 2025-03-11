#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import random
import logging

logger = logging.getLogger(__name__)

class HDF5SegmentationDataset(Dataset):
    """Dataset for loading preprocessed nanopore signal data from HDF5 files."""
    
    def __init__(
        self,
        hdf5_path: str,
        transform=None,
        augment: bool = False,
        augment_params: Optional[Dict] = None,
        cache_size: int = 1000,
        train: bool = True,
        val_split: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize the HDF5 dataset.
        
        Args:
            hdf5_path: Path to the HDF5 file containing preprocessed data
            transform: Optional transform to apply to samples
            augment: Whether to apply data augmentation
            augment_params: Parameters for data augmentation
            cache_size: Number of samples to cache in memory
            train: Whether this dataset is for training (vs. validation)
            val_split: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.augment = augment
        self.augment_params = augment_params or {}
        self.cache_size = cache_size
        self.train = train
        self.val_split = val_split
        
        # Open HDF5 file in read mode
        self.h5_file = h5py.File(hdf5_path, 'r')
        
        # Get dataset parameters
        self.chunk_size = self.h5_file['metadata']['parameters'].attrs['chunk_size']
        self.chunk_stride = self.h5_file['metadata']['parameters'].attrs['chunk_stride']
        
        # Get all read IDs
        all_read_ids = []
        
        # Check if read_ids are stored in metadata
        if 'metadata' in self.h5_file and 'read_ids' in self.h5_file['metadata']:
            # Read from metadata
            read_ids_dataset = self.h5_file['metadata']['read_ids'][:]
            all_read_ids = [r_id.decode('utf-8') if isinstance(r_id, bytes) else r_id 
                           for r_id in read_ids_dataset]
        else:
            # Fallback to keys in signals group
            all_read_ids = list(self.h5_file['signals'].keys())
        
        # Split into train and validation sets
        random.seed(seed)
        random.shuffle(all_read_ids)
        
        val_size = int(len(all_read_ids) * val_split)
        if train:
            self.read_ids = all_read_ids[val_size:]
        else:
            self.read_ids = all_read_ids[:val_size]
        
        # Build index mapping
        self.index_map = []  # (read_id, chunk_idx) pairs
        for read_id in self.read_ids:
            # Convert to string if it's bytes
            if isinstance(read_id, bytes):
                read_id = read_id.decode('utf-8')
                
            if read_id in self.h5_file['signals']:
                n_chunks = len(self.h5_file['signals'][read_id]['chunks'])
                for i in range(n_chunks):
                    self.index_map.append((read_id, i))
        
        logger.info(f"Loaded {len(self.index_map)} chunks from {len(self.read_ids)} reads")
        
        # Initialize cache
        self.cache = {}
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        read_id, chunk_idx = self.index_map[idx]
        
        # Check if in cache
        cache_key = (read_id, chunk_idx)
        if cache_key in self.cache:
            signal, label = self.cache[cache_key]
        else:
            # Convert read_id to string if it's bytes
            if isinstance(read_id, bytes):
                read_id = read_id.decode('utf-8')
                
            # Load from HDF5
            signal = self.h5_file['signals'][read_id]['chunks'][chunk_idx]
            label = self.h5_file['labels'][read_id]['labels'][chunk_idx]
            
            # Convert to torch tensors
            signal = torch.from_numpy(signal.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
            
            # Apply transform if provided
            if self.transform:
                signal, label = self.transform(signal, label)
            
            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove random item if cache is full
                remove_key = random.choice(list(self.cache.keys()))
                del self.cache[remove_key]
            
            self.cache[cache_key] = (signal, label)
        
        # Apply data augmentation during training
        if self.train and self.augment:
            signal = self._augment_signal(signal)
        
        return signal.unsqueeze(0), label  # Add channel dimension
    
    def _augment_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to the signal."""
        # Scale augmentation
        if random.random() < self.augment_params.get('scale_prob', 0.3):
            scale_factor = random.uniform(
                self.augment_params.get('scale_min', 0.8),
                self.augment_params.get('scale_max', 1.2)
            )
            signal = signal * scale_factor
        
        # Noise augmentation
        if random.random() < self.augment_params.get('noise_prob', 0.3):
            noise_level = random.uniform(
                self.augment_params.get('noise_min', 0.01),
                self.augment_params.get('noise_max', 0.1)
            )
            noise = torch.randn_like(signal) * noise_level
            signal = signal + noise
        
        # Time shift augmentation
        if random.random() < self.augment_params.get('shift_prob', 0.3):
            max_shift = self.augment_params.get('max_shift', 100)
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                signal = torch.cat([torch.zeros(shift), signal[:-shift]])
            elif shift < 0:
                signal = torch.cat([signal[-shift:], torch.zeros(-shift)])
        
        return signal
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
    
    def __del__(self):
        """Clean up when the object is deleted."""
        self.close()


def create_train_val_dataloaders(
    hdf5_path: str,
    batch_size: int = 32,
    augment: bool = True,
    augment_params: Optional[Dict] = None,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        batch_size: Batch size for the DataLoaders
        augment: Whether to apply data augmentation to the training set
        augment_params: Parameters for data augmentation
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for DataLoader
        seed: Random seed for reproducibility
        
    Returns:
        A tuple of (train_dataloader, val_dataloader)
    """
    # Training dataset
    train_dataset = HDF5SegmentationDataset(
        hdf5_path=hdf5_path,
        augment=augment,
        augment_params=augment_params,
        train=True,
        val_split=val_split,
        seed=seed
    )
    
    # Validation dataset
    val_dataset = HDF5SegmentationDataset(
        hdf5_path=hdf5_path,
        augment=False,
        train=False,
        val_split=val_split,
        seed=seed
    )
    
    # Create DataLoaders
    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader