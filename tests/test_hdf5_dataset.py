import os
import tempfile
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch

from segmentit.data.hdf5_dataset import HDF5SegmentationDataset, create_train_val_dataloaders

@pytest.fixture
def sample_hdf5_file():
    """Create a sample HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp:
        # Create a temporary HDF5 file
        with h5py.File(temp.name, 'w') as f:
            # Create dataset groups
            signals_group = f.create_group('signals')
            labels_group = f.create_group('labels')
            metadata_group = f.create_group('metadata')
            
            # Store dataset parameters
            params = metadata_group.create_group('parameters')
            params.attrs['chunk_size'] = 4000
            params.attrs['chunk_stride'] = 2000
            
            # Create 5 test reads with 3 chunks each
            for i in range(5):
                read_id = f'read_{i}'
                
                # Create signal chunks
                chunks = np.random.randn(3, 4000).astype(np.float32)
                chunk_indices = np.array([0, 1, 2])
                
                # Create labels
                labels = np.zeros((3, 4000), dtype=np.float32)
                # Add some random segment boundaries
                for j in range(3):
                    boundaries = np.random.choice(4000, size=4, replace=False)
                    labels[j, boundaries] = 1.0
                
                # Store in HDF5
                read_group = signals_group.create_group(read_id)
                read_group.create_dataset('chunks', data=chunks)
                read_group.create_dataset('indices', data=chunk_indices)
                
                label_group = labels_group.create_group(read_id)
                label_group.create_dataset('labels', data=labels)
    
    yield temp.name
    # Clean up
    os.unlink(temp.name)

class TestHDF5Dataset:
    
    def test_dataset_initialization(self, sample_hdf5_file):
        """Test dataset initialization and basic properties."""
        # Create dataset
        dataset = HDF5SegmentationDataset(sample_hdf5_file)
        
        # Check length (should be number of reads * chunks per read * train split)
        expected_len = int(5 * 3 * (1 - 0.1))  # 5 reads, 3 chunks, 90% train
        assert len(dataset) == expected_len
        
        # Check parameters
        assert dataset.chunk_size == 4000
        assert dataset.chunk_stride == 2000
        
        # Clean up
        dataset.close()
    
    def test_getitem(self, sample_hdf5_file):
        """Test retrieving items from the dataset."""
        dataset = HDF5SegmentationDataset(sample_hdf5_file)
        
        # Get first item
        signal, label = dataset[0]
        
        # Check shapes
        assert signal.shape == (1, 4000)  # (channels, length)
        assert label.shape == (4000,)
        
        # Check types
        assert signal.dtype == torch.float32
        assert label.dtype == torch.float32
        
        # Clean up
        dataset.close()
    
    def test_caching(self, sample_hdf5_file):
        """Test that caching works."""
        dataset = HDF5SegmentationDataset(sample_hdf5_file, cache_size=5)
        
        # Access a few items to populate the cache
        for i in range(3):
            _ = dataset[i]
        
        # Check that items are in the cache
        assert len(dataset.cache) == 3
        
        # Access the same items again
        for i in range(3):
            _ = dataset[i]
        
        # Cache size should still be 3
        assert len(dataset.cache) == 3
        
        # Clean up
        dataset.close()
    
    def test_train_val_split(self, sample_hdf5_file):
        """Test that train/val split works correctly."""
        # Create training and validation datasets
        train_dataset = HDF5SegmentationDataset(
            sample_hdf5_file, train=True, val_split=0.2)
        val_dataset = HDF5SegmentationDataset(
            sample_hdf5_file, train=False, val_split=0.2)
        
        # Check that they have the right sizes
        total_chunks = 5 * 3  # 5 reads, 3 chunks each
        expected_train = int(total_chunks * 0.8)
        expected_val = int(total_chunks * 0.2)
        
        assert len(train_dataset) + len(val_dataset) <= total_chunks
        assert abs(len(train_dataset) - expected_train) <= 1
        assert abs(len(val_dataset) - expected_val) <= 1
        
        # Make sure they don't overlap
        train_indices = set(train_dataset.index_map)
        val_indices = set(val_dataset.index_map)
        assert len(train_indices.intersection(val_indices)) == 0
        
        # Clean up
        train_dataset.close()
        val_dataset.close()
    
    def test_dataloader_creation(self, sample_hdf5_file):
        """Test creating DataLoaders."""
        train_loader, val_loader = create_train_val_dataloaders(
            sample_hdf5_file, batch_size=2, num_workers=0)
        
        # Check that we can iterate through the DataLoader
        for signals, labels in train_loader:
            assert signals.shape[0] <= 2  # Batch size
            assert signals.shape[1] == 1  # Channels
            assert signals.shape[2] == 4000  # Signal length
            assert labels.shape[0] <= 2  # Batch size
            assert labels.shape[1] == 4000  # Label length
            break
        
        # Clean up
        train_loader.dataset.close()
        val_loader.dataset.close()
    
    def test_augmentation(self, sample_hdf5_file):
        """Test that augmentation works."""
        # Create dataset with augmentation
        augment_params = {
            'scale_prob': 1.0,  # Always apply scaling
            'scale_min': 0.5,
            'scale_max': 1.5,
            'noise_prob': 0.0,  # No noise
            'shift_prob': 0.0   # No shift
        }
        
        dataset = HDF5SegmentationDataset(
            sample_hdf5_file, 
            augment=True, 
            augment_params=augment_params
        )
        
        # Get the same item twice to check if augmentation is different
        signal1, _ = dataset[0]
        signal2, _ = dataset[0]
        
        # Since we're applying random scaling, the signals should be different
        assert not torch.allclose(signal1, signal2)
        
        # Clean up
        dataset.close()