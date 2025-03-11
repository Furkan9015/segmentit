import os
import tempfile
from pathlib import Path
import h5py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from scripts.prepare_data import process_read, prepare_signal_chunk
from segmentit.utils.signal_processing import normalize_signal, median_filter

class TestDataPreparation:
    
    def test_prepare_signal_chunk(self):
        # Test with a simple signal and no labels
        signal = np.random.randn(10000)
        chunk = prepare_signal_chunk(signal, 0, 4000, 2000)
        
        assert chunk.shape == (4000,)
        assert chunk.mean() < 0.1  # Should be approximately zero after normalization
        
        # Test with labels
        labels = [(500, 1000), (2500, 3000)]
        chunk, label_chunk = prepare_signal_chunk(signal, 0, 4000, 2000, labels)
        
        assert chunk.shape == (4000,)
        assert label_chunk.shape == (4000,)
        assert label_chunk[500] == 1  # Start boundary
        assert label_chunk[1000] == 1  # End boundary
        assert label_chunk[2500] == 1  # Start boundary
        assert label_chunk[3000] == 1  # End boundary
        assert np.sum(label_chunk) == 4  # Only 4 boundary points
    
    def test_process_read(self):
        # Create a test signal and labels
        signal = np.random.randn(20000)
        labels = [(1000, 1500), (6000, 7000), (12000, 13000)]
        
        # Process the read
        result = process_read('test_read', signal, labels, 4000, 2000, 1)
        
        # Check results
        assert result is not None
        read_id, chunks, chunk_labels, chunk_indices = result
        
        assert read_id == 'test_read'
        assert chunks.shape[0] == chunk_labels.shape[0]  # Same number of chunks and labels
        assert chunks.shape[1] == 4000  # Chunk size
        assert chunk_labels.shape[1] == 4000  # Label size
        
        # Test with minimum segments filter
        result = process_read('test_read', signal, labels, 4000, 2000, 2)
        assert result is None or result[1].shape[0] < chunks.shape[0]  # Fewer or no chunks
        
    @pytest.mark.parametrize("compression", ["gzip", "lzf", None])
    def test_hdf5_write_performance(self, compression):
        # Create test data
        chunk_count = 10
        chunks = np.random.randn(chunk_count, 4000)
        labels = np.random.randint(0, 2, (chunk_count, 4000)).astype(np.float32)
        indices = np.arange(chunk_count)
        
        # Write to HDF5
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp:
            with h5py.File(temp.name, 'w') as f:
                # Create dataset groups
                signals_group = f.create_group('signals')
                labels_group = f.create_group('labels')
                
                # Store in HDF5
                read_group = signals_group.create_group('test_read')
                read_group.create_dataset('chunks', data=chunks, compression=compression)
                read_group.create_dataset('indices', data=indices, compression=compression)
                
                label_group = labels_group.create_group('test_read')
                label_group.create_dataset('labels', data=labels, compression=compression)
            
            # Read and verify
            with h5py.File(temp.name, 'r') as f:
                read_chunks = f['signals']['test_read']['chunks'][:]
                read_labels = f['labels']['test_read']['labels'][:]
                
                np.testing.assert_array_equal(chunks, read_chunks)
                np.testing.assert_array_equal(labels, read_labels)