"""Inference module for nanopore segmentation models."""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging
import time
from tqdm import tqdm

from segmentit.data.fast5_reader import Fast5Reader
from segmentit.utils.signal_processing import normalize_signal, median_filter

logger = logging.getLogger(__name__)

class SegmentationInference:
    """Class for inference with segmentation models."""
    
    def __init__(
        self,
        model_path: Path,
        device: Optional[torch.device] = None,
        use_onnx: bool = False,
        chunk_size: int = 4000,
        stride: int = 2000,
        normalize: bool = True,
        filter_signal: bool = True,
        threshold: float = 0.5,
    ):
        """Initialize inference.
        
        Args:
            model_path: Path to model file (PyTorch checkpoint or ONNX)
            device: Device to use for PyTorch model
            use_onnx: Whether to use ONNX runtime for inference
            chunk_size: Size of chunks to process
            stride: Stride for sliding window
            normalize: Whether to normalize signal
            filter_signal: Whether to apply median filtering
            threshold: Threshold for boundary detection
        """
        self.model_path = Path(model_path)
        self.use_onnx = use_onnx
        self.chunk_size = chunk_size
        self.stride = stride
        self.normalize = normalize
        self.filter_signal = filter_signal
        self.threshold = threshold
        
        # Setup device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
            
        logger.info(f"Loaded segmentation model from {model_path}")
    
    def _load_pytorch_model(self) -> None:
        """Load PyTorch model."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Check if it's a checkpoint or a model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            from segmentit.models.segmentation_model import SegmentationModel, EfficientUNet
            
            # Determine model type from checkpoint if available
            if 'model_type' in checkpoint:
                model_type = checkpoint['model_type']
                if model_type == 'SegmentationModel':
                    self.model = SegmentationModel()
                else:
                    self.model = EfficientUNet()
            else:
                # Try to infer model type from state dict keys
                state_dict = checkpoint['model_state_dict']
                if any('upconv_blocks' in key for key in state_dict.keys()):
                    self.model = EfficientUNet()
                else:
                    self.model = SegmentationModel()
                    
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume it's a direct model
            self.model = checkpoint
            
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_onnx_model(self) -> None:
        """Load ONNX model for inference."""
        # Use CPU execution provider for stability
        self.onnx_session = ort.InferenceSession(
            str(self.model_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Get input and output names
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name
    
    def process_signal(
        self, signal: np.ndarray, read_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Process a signal and detect segment boundaries.
        
        Args:
            signal: Input signal
            read_id: Read ID for logging
            
        Returns:
            Dictionary with boundary predictions and positions
        """
        if len(signal) < self.chunk_size:
            # Pad signal if it's shorter than chunk_size
            padding = np.zeros(self.chunk_size - len(signal))
            signal = np.concatenate([signal, padding])
        
        # Apply preprocessing
        if self.filter_signal:
            signal = median_filter(signal, window_size=5)
        
        # Process signal in chunks with overlap
        chunks = []
        start_indices = []
        
        for start_idx in range(0, len(signal) - self.chunk_size + 1, self.stride):
            end_idx = start_idx + self.chunk_size
            chunk = signal[start_idx:end_idx].copy()
            
            if self.normalize:
                chunk = normalize_signal(chunk)
                
            chunks.append(chunk)
            start_indices.append(start_idx)
        
        # If there's a remaining part at the end
        if len(signal) > start_indices[-1] + self.chunk_size:
            start_idx = len(signal) - self.chunk_size
            chunk = signal[start_idx:].copy()
            
            if self.normalize:
                chunk = normalize_signal(chunk)
                
            chunks.append(chunk)
            start_indices.append(start_idx)
        
        # Process all chunks
        all_boundaries = np.zeros(len(signal), dtype=np.float32)
        
        # Process chunks in batches for efficiency
        batch_size = 32
        num_chunks = len(chunks)
        num_batches = (num_chunks + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_batch = batch_idx * batch_size
            end_batch = min((batch_idx + 1) * batch_size, num_chunks)
            
            batch_chunks = chunks[start_batch:end_batch]
            batch_indices = start_indices[start_batch:end_batch]
            
            # Convert to appropriate format for inference
            if self.use_onnx:
                batch_array = np.array(batch_chunks, dtype=np.float32)
                batch_array = batch_array.reshape(-1, 1, self.chunk_size)
                
                # Run ONNX inference
                outputs = self.onnx_session.run(
                    [self.output_name], {self.input_name: batch_array}
                )[0]
            else:
                batch_tensor = torch.tensor(batch_chunks, dtype=torch.float32, device=self.device)
                batch_tensor = batch_tensor.unsqueeze(1)  # Add channel dimension
                
                # Run PyTorch inference
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    outputs = outputs.cpu().numpy()
            
            # Fill in the results
            for i, start_idx in enumerate(batch_indices[0:end_batch-start_batch]):
                output = outputs[i].squeeze()
                
                # Use weighted average for overlapping regions
                weight = np.ones_like(output)
                
                # Down-weight the edges of each chunk to handle overlap better
                fade_len = min(self.stride // 2, 200)
                if fade_len > 0:
                    # Linear fade at the beginning and end of chunk
                    weight[:fade_len] = np.linspace(0.5, 1.0, fade_len)
                    weight[-fade_len:] = np.linspace(1.0, 0.5, fade_len)
                
                # Update results with weighted average
                all_boundaries[start_idx:start_idx+self.chunk_size] += output * weight
                
        # Apply threshold to get binary boundaries
        binary_boundaries = (all_boundaries > self.threshold).astype(np.int32)
        
        # Get boundary positions
        boundary_positions = np.where(binary_boundaries == 1)[0]
        
        return {
            'all_boundaries': all_boundaries,
            'binary_boundaries': binary_boundaries,
            'boundary_positions': boundary_positions
        }
    
    def process_read(
        self, fast5_reader: Fast5Reader, read_id: str
    ) -> Optional[Dict[str, Any]]:
        """Process a single read and detect segment boundaries.
        
        Args:
            fast5_reader: Fast5Reader instance
            read_id: Read ID to process
            
        Returns:
            Dictionary with results or None if read not found
        """
        # Get signal for the read
        signal = fast5_reader.get_signal_by_read_id(read_id)
        if signal is None:
            logger.warning(f"Signal not found for read_id {read_id}")
            return None
        
        # Process signal
        results = self.process_signal(signal, read_id)
        
        # Add metadata
        results['read_id'] = read_id
        results['signal_length'] = len(signal)
        
        return results
    
    def process_multiple_reads(
        self, fast5_reader: Fast5Reader, read_ids: List[str], max_workers: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        """Process multiple reads in parallel.
        
        Args:
            fast5_reader: Fast5Reader instance
            read_ids: List of read IDs to process
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            Dictionary mapping read IDs to results
        """
        from concurrent.futures import ThreadPoolExecutor
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_read, fast5_reader, read_id): read_id 
                for read_id in read_ids
            }
            
            for future in tqdm(futures, desc="Processing reads", total=len(read_ids)):
                read_id = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[read_id] = result
                except Exception as e:
                    logger.error(f"Error processing read {read_id}: {e}")
        
        return results
    
    def benchmark_speed(self, signal_length: int = 400000, num_runs: int = 10) -> float:
        """Benchmark inference speed.
        
        Args:
            signal_length: Length of signal to benchmark with
            num_runs: Number of runs
            
        Returns:
            Average processing time in seconds
        """
        # Generate random signal
        np.random.seed(42)
        signal = np.random.randn(signal_length).astype(np.float32)
        
        # Warm-up run
        _ = self.process_signal(signal)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = self.process_signal(signal)
            
        end_time = time.time()
        
        # Calculate average time
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        
        # Calculate throughput
        throughput = signal_length / avg_time
        
        logger.info(f"Average processing time: {avg_time:.4f} seconds for {signal_length} datapoints")
        logger.info(f"Throughput: {throughput:.2f} datapoints/second")
        
        return avg_time