"""Trainer module for nanopore segmentation models."""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import os
import time
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from segmentit.models.segmentation_model import SegmentationModel, EfficientUNet
from segmentit.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for nanopore segmentation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        log_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        metrics_fn: Optional[Callable] = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            optimizer: Optimizer to use (default: Adam)
            criterion: Loss function (default: BCELoss)
            scheduler: Learning rate scheduler (optional)
            device: Device to use (default: cuda if available, else cpu)
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
            metrics_fn: Function to compute metrics (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        
        # Set default optimizer if not provided
        self.optimizer = optimizer if optimizer is not None else optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=1e-5
        )
        
        # Set default loss function if not provided
        self.criterion = criterion if criterion is not None else nn.BCELoss()
        
        # Set scheduler
        self.scheduler = scheduler
        
        # Set metrics function
        self.metrics_fn = metrics_fn if metrics_fn is not None else compute_metrics
        
        # Set up directories
        self.log_dir = Path(log_dir) if log_dir is not None else Path('./logs')
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path('./checkpoints')
        
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        # Best validation loss for model saving
        self.best_val_loss = float('inf')
        
    def train(
        self,
        num_epochs: int,
        patience: int = 5,
        eval_interval: int = 1,
        save_best_only: bool = True,
        early_stopping: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
            patience: Number of epochs to wait before early stopping
            eval_interval: Evaluate model every N epochs
            save_best_only: Save only the best model
            early_stopping: Whether to use early stopping
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # For early stopping
        no_improvement = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self._train_epoch()
            
            # Log training loss
            self.history['train_loss'].append(train_loss)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            
            # Evaluate model if validation set is available
            if self.val_loader is not None and (epoch + 1) % eval_interval == 0:
                val_loss, metrics = self._evaluate()
                
                # Log validation loss and metrics
                self.history['val_loss'].append(val_loss)
                self.history['metrics'].append(metrics)
                
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                for metric_name, metric_value in metrics.items():
                    self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                
                # Update learning rate scheduler if available
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    no_improvement = 0
                    
                    if save_best_only:
                        self._save_checkpoint(epoch, val_loss, metrics, is_best=True)
                else:
                    no_improvement += 1
                    if early_stopping and no_improvement >= patience:
                        logger.info(f"Early stopping after {epoch + 1} epochs")
                        break
                
                # Print progress
                time_taken = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"F1: {metrics.get('f1', 0):.4f}, "
                    f"Time: {time_taken:.2f}s"
                )
            else:
                # Print progress without validation
                time_taken = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Time: {time_taken:.2f}s"
                )
                
                # Update learning rate scheduler if available
                if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
                    
            # Save checkpoint periodically if not save_best_only
            if not save_best_only and (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, val_loss if self.val_loader else train_loss, 
                                     metrics if self.val_loader else {}, is_best=False)
        
        # Save final model if not saved already
        if not save_best_only:
            self._save_checkpoint(
                num_epochs - 1,
                self.history['val_loss'][-1] if self.val_loader and self.history['val_loss'] else train_loss,
                self.history['metrics'][-1] if self.val_loader and self.history['metrics'] else {},
                is_best=False
            )
            
        # Close TensorBoard writer
        self.writer.close()
        
        return self.history
    
    def _train_epoch(self) -> float:
        """Train the model for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        # Use tqdm for progress bar
        with tqdm(total=len(self.train_loader), desc='Training', leave=False) as pbar:
            for batch in self.train_loader:
                # Move data to device
                signals = batch['signal'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(signals)
                
                # Reshape outputs to match labels if needed
                if outputs.shape != labels.shape:
                    outputs = outputs.view(labels.shape)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss
    
    def _evaluate(self) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on the validation set.
        
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                signals = batch['signal'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(signals)
                
                # Reshape outputs to match labels if needed
                if outputs.shape != labels.shape:
                    outputs = outputs.view(labels.shape)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Update statistics
                total_loss += loss.item()
                
                # Store outputs and labels for metric calculation
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)
        
        # Concatenate all outputs and labels
        all_outputs = np.concatenate([output.flatten() for output in all_outputs])
        all_labels = np.concatenate([label.flatten() for label in all_labels])
        
        # Apply threshold to get binary predictions
        threshold = 0.5
        binary_preds = (all_outputs > threshold).astype(np.int32)
        
        # Calculate metrics
        metrics = self.metrics_fn(all_labels, binary_preds, all_outputs)
        
        return avg_loss, metrics
    
    def _save_checkpoint(
        self, epoch: int, loss: float, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Validation loss
            metrics: Evaluation metrics
            is_best: Whether this is the best model so far
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
            logger.info(f"Saved best model checkpoint (Epoch {epoch+1}, Loss: {loss:.4f})")
        else:
            torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
            logger.info(f"Saved checkpoint for epoch {epoch+1}")
    
    def export_to_onnx(self, output_path: Path, dynamic_axes: bool = True) -> None:
        """Export the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            dynamic_axes: Whether to use dynamic axes for variable batch size and sequence length
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create dummy input for ONNX export
        batch_size = 1
        seq_len = 4000
        dummy_input = torch.randn(batch_size, 1, seq_len, device=self.device)
        
        # Define dynamic axes if needed
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size', 2: 'sequence_length'},
                'output': {0: 'batch_size', 2: 'sequence_length'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict
        )
        
        logger.info(f"Exported model to ONNX: {output_path}")
        
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Set best validation loss
        self.best_val_loss = checkpoint['loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (Epoch {checkpoint['epoch']+1})")
        
    def get_prediction_time(self, batch_size: int = 1, seq_len: int = 4000, num_runs: int = 100) -> float:
        """Measure inference time for a single batch.
        
        Args:
            batch_size: Batch size to test
            seq_len: Sequence length to test
            num_runs: Number of inference runs to average
            
        Returns:
            Average inference time in milliseconds
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 1, seq_len, device=self.device)
        
        # Warm-up runs
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Measure inference time
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(dummy_input)
        end_time = time.time()
        
        # Calculate average time in milliseconds
        avg_time = (end_time - start_time) / num_runs * 1000
        
        return avg_time