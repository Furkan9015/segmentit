"""Fast5 file reader for nanopore signal data."""

from typing import Dict, List, Tuple, Optional, Any
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Fast5Reader:
    """Reader class for extracting raw signal from Fast5 files."""
    
    def __init__(self, fast5_dir: Path):
        """Initialize the Fast5Reader.
        
        Args:
            fast5_dir: Directory containing Fast5 files
        """
        self.fast5_dir = Path(fast5_dir)
        if not self.fast5_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.fast5_dir}")
            
        self.fast5_files = list(self.fast5_dir.glob("*.fast5"))
        logger.info(f"Found {len(self.fast5_files)} Fast5 files in {self.fast5_dir}")
        
        # Cache for read_id to file mapping
        self._read_id_to_file: Dict[str, Path] = {}
        self._build_read_id_cache()
        
    def _build_read_id_cache(self) -> None:
        """Build a cache mapping read_ids to their Fast5 files."""
        for file_path in self.fast5_files:
            try:
                with h5py.File(file_path, 'r') as f5:
                    for read_group in f5:
                        # Extract read_id from the read group
                        try:
                            read_id = f5[read_group].attrs.get('read_id')
                            if read_id is None:
                                # For multi-read files with different structure
                                read_id = f5[read_group].attrs.get('read_id')
                            if isinstance(read_id, bytes):
                                read_id = read_id.decode('utf-8')
                            
                            if read_id:
                                self._read_id_to_file[read_id] = file_path
                        except Exception as e:
                            logger.warning(f"Failed to get read_id from {file_path}/{read_group}: {e}")
            except Exception as e:
                logger.warning(f"Failed to open {file_path}: {e}")
                
        logger.info(f"Cached {len(self._read_id_to_file)} read_ids")
    
    def get_signal_by_read_id(self, read_id: str) -> Optional[np.ndarray]:
        """Get raw signal data for a specific read_id.
        
        Args:
            read_id: The read ID to retrieve signal for
            
        Returns:
            Raw signal as numpy array or None if not found
        """
        if read_id not in self._read_id_to_file:
            logger.warning(f"Read ID {read_id} not found in cache")
            return None
            
        file_path = self._read_id_to_file[read_id]
        try:
            with h5py.File(file_path, 'r') as f5:
                for read_group in f5:
                    current_id = f5[read_group].attrs.get('read_id')
                    if isinstance(current_id, bytes):
                        current_id = current_id.decode('utf-8')
                        
                    if current_id == read_id:
                        # Signal might be in different locations based on file format
                        try:
                            # ONT format
                            signal = f5[f"{read_group}/Raw/Signal"][:]
                            return signal
                        except KeyError:
                            # Try alternative path for newer versions
                            try:
                                signal = f5[f"{read_group}/Raw"][:]
                                return signal
                            except KeyError:
                                logger.warning(f"Could not find signal data for {read_id}")
                                return None
        except Exception as e:
            logger.error(f"Error reading file {file_path} for read_id {read_id}: {e}")
            return None
            
        return None
    
    def get_chunk_from_signal(self, signal: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """Extract a chunk from the signal based on start and end indices.
        
        Args:
            signal: The full signal array
            start_idx: Start index
            end_idx: End index
            
        Returns:
            Chunk of signal
        """
        if start_idx >= len(signal) or end_idx > len(signal):
            logger.warning(f"Invalid indices: start={start_idx}, end={end_idx}, signal_length={len(signal)}")
            # Return empty array or handle error appropriately
            return np.array([])
            
        return signal[start_idx:end_idx]
        
    def get_all_read_ids(self) -> List[str]:
        """Get list of all available read IDs.
        
        Returns:
            List of read IDs
        """
        return list(self._read_id_to_file.keys())