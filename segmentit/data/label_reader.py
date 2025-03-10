"""Label reader for nanopore signal segmentation."""

from typing import Dict, List, Tuple, Set, Optional
import pyarrow as pa
import pyarrow.csv as csv
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LabelReader:
    """Reader class for segment labels from TSV files using PyArrow for efficiency."""
    
    def __init__(self, tsv_path: Path):
        """Initialize the LabelReader.
        
        Args:
            tsv_path: Path to the TSV file containing segmentation labels
        """
        self.tsv_path = Path(tsv_path)
        if not self.tsv_path.exists():
            raise FileNotFoundError(f"TSV file not found: {self.tsv_path}")
        
        # Use PyArrow for efficient CSV/TSV reading
        read_options = csv.ReadOptions(use_threads=True)
        parse_options = csv.ParseOptions(delimiter='\t')
        
        # Load data using PyArrow
        table = csv.read_csv(str(self.tsv_path), read_options=read_options, parse_options=parse_options)
        self.labels_df = table.to_pandas()
        logger.info(f"Loaded {len(self.labels_df)} segment labels from {self.tsv_path}")
        
        # Validate expected columns
        expected_columns = {'read_id', 'kmer_idx', 'start_raw_idx', 'end_raw_idx'}
        if not expected_columns.issubset(set(self.labels_df.columns)):
            missing = expected_columns - set(self.labels_df.columns)
            raise ValueError(f"Missing required columns in TSV: {missing}")
        
        # Convert string columns to categorical for memory efficiency
        if self.labels_df['read_id'].dtype == 'object':
            self.labels_df['read_id'] = self.labels_df['read_id'].astype('category')
        
        # Ensure index columns are integers
        self.labels_df['start_raw_idx'] = self.labels_df['start_raw_idx'].astype(np.int32)
        self.labels_df['end_raw_idx'] = self.labels_df['end_raw_idx'].astype(np.int32)
        self.labels_df['kmer_idx'] = self.labels_df['kmer_idx'].astype(np.int32)
        
        # Create a dictionary for faster access to segments by read_id
        self._read_segments: Dict[str, pd.DataFrame] = {}
        self._organize_by_read_id()
        
    def _organize_by_read_id(self) -> None:
        """Organize segments by read_id for faster access."""
        # Group by read_id and sort by start_raw_idx for efficient lookups
        for read_id, group in self.labels_df.groupby('read_id', observed=True):
            # Sort within each group and reset index for faster access
            self._read_segments[read_id] = group.sort_values('start_raw_idx').reset_index(drop=True)
    
    def get_read_ids(self) -> Set[str]:
        """Get all read IDs in the labels file.
        
        Returns:
            Set of read IDs
        """
        return set(self._read_segments.keys())
    
    def get_segments_for_read(self, read_id: str) -> Optional[pd.DataFrame]:
        """Get segments for a specific read.
        
        Args:
            read_id: Read ID to get segments for
            
        Returns:
            DataFrame with segments or None if read_id not found
        """
        return self._read_segments.get(read_id)
    
    def get_segments_in_range(self, read_id: str, start_idx: int, end_idx: int) -> Optional[pd.DataFrame]:
        """Get segments that overlap with a specific range.
        
        Args:
            read_id: Read ID to get segments for
            start_idx: Start index of the range
            end_idx: End index of the range
            
        Returns:
            DataFrame with segments that overlap with the range or None if read_id not found
        """
        segments = self.get_segments_for_read(read_id)
        if segments is None:
            return None
        
        # Use vectorized operations for speed
        # Find segments that overlap with the range (any overlap)
        mask = (
            ((segments['start_raw_idx'] >= start_idx) & (segments['start_raw_idx'] < end_idx)) |
            ((segments['end_raw_idx'] > start_idx) & (segments['end_raw_idx'] <= end_idx)) |
            ((segments['start_raw_idx'] <= start_idx) & (segments['end_raw_idx'] >= end_idx))
        )
        
        return segments[mask]
    
    def generate_segmentation_mask(self, read_id: str, start_idx: int, end_idx: int) -> np.ndarray:
        """Generate a binary mask for segment boundaries within a specified range.
        
        Args:
            read_id: Read ID to generate mask for
            start_idx: Start index of the range
            end_idx: End index of the range
            
        Returns:
            Binary mask where 1 indicates a segment boundary
        """
        segments = self.get_segments_in_range(read_id, start_idx, end_idx)
        length = end_idx - start_idx
        
        if segments is None or len(segments) == 0:
            return np.zeros(length, dtype=np.int8)
        
        mask = np.zeros(length, dtype=np.int8)
        
        # Mark segment boundaries with 1, adjusting for the start_idx offset
        for _, row in segments.iterrows():
            seg_start = row['start_raw_idx']
            seg_end = row['end_raw_idx']
            
            # Adjust indices relative to start_idx
            if seg_start >= start_idx and seg_start < end_idx:
                mask[seg_start - start_idx] = 1
            if seg_end > start_idx and seg_end < end_idx:
                mask[seg_end - start_idx] = 1
                
        return mask
    
    def generate_boundary_labels(self, read_id: str, start_idx: int, end_idx: int) -> np.ndarray:
        """Generate boundary labels for segmentation, optimized for model training.
        
        Args:
            read_id: Read ID to generate labels for
            start_idx: Start index of the range
            end_idx: End index of the range
            
        Returns:
            Array with boundary labels (1 at segment boundaries, 0 elsewhere)
        """
        segments = self.get_segments_in_range(read_id, start_idx, end_idx)
        length = end_idx - start_idx
        
        if segments is None or len(segments) == 0:
            return np.zeros(length, dtype=np.int8)
        
        # Pre-allocate with zeros
        boundaries = np.zeros(length, dtype=np.int8)
        
        # Use numpy operations for efficiency
        # Extract all start and end indices
        starts = segments['start_raw_idx'].values
        ends = segments['end_raw_idx'].values
        
        # Filter boundaries that are within our range and adjust to relative positions
        valid_starts = starts[(starts >= start_idx) & (starts < end_idx)] - start_idx
        valid_ends = ends[(ends > start_idx) & (ends < end_idx)] - start_idx
        
        # Set boundaries
        boundaries[valid_starts] = 1
        boundaries[valid_ends] = 1
        
        return boundaries