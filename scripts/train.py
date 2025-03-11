#!/usr/bin/env python
"""Training script for nanopore segmentation models."""

import argparse
import yaml
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from pathlib import Path
import random
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentit.data.fast5_reader import Fast5Reader
from segmentit.data.label_reader import LabelReader
from segmentit.data.dataset import NanoporeSegmentationDataset
from segmentit.data.hdf5_dataset import HDF5SegmentationDataset, create_train_val_dataloaders
from segmentit.models.segmentation_model import SegmentationModel, EfficientUNet
from segmentit.models.trainer import Trainer
from segmentit.utils.metrics import plot_loss_curves, plot_metric_curves

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger('train')

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_loss_function(config):
    """Create loss function based on config.
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss function
    """
    loss_type = config.get('type', 'bce')
    
    if loss_type == 'bce':
        if 'pos_weight' in config:
            pos_weight = torch.tensor([config['pos_weight']])
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.BCELoss()
        
    elif loss_type == 'focal':
        from segmentit.utils.losses import FocalLoss
        alpha = config.get('focal_alpha', 0.25)
        gamma = config.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
        
    elif loss_type == 'dice':
        from segmentit.utils.losses import DiceLoss
        return DiceLoss()
        
    elif loss_type == 'weighted_bce':
        from segmentit.utils.losses import WeightedBCELoss
        pos_weight = config.get('pos_weight', 5.0)
        return WeightedBCELoss(pos_weight=pos_weight)
        
    else:
        logger.warning(f"Unknown loss type: {loss_type}, using BCE")
        return nn.BCELoss()

def create_scheduler(optimizer, config, epochs):
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        config: Scheduler configuration
        epochs: Number of epochs
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('type', 'step')
    
    if scheduler_type == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_type == 'cosine':
        eta_min = config.get('min_lr', 0.00001)
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        
    elif scheduler_type == 'reduce':
        factor = config.get('factor', 0.5)
        patience = config.get('patience', 5)
        min_lr = config.get('min_lr', 0.00001)
        return ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, verbose=True, min_lr=min_lr
        )
        
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using StepLR")
        return StepLR(optimizer, step_size=10, gamma=0.5)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train a nanopore segmentation model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with limited data')
    parser.add_argument('--hdf5', type=str, help='Path to HDF5 dataset file (skips Fast5/TSV loading if provided)')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Debug mode
    if args.debug:
        logger.info("Running in debug mode with limited data")
        config['data']['batch_size'] = 8
        config['training']['num_epochs'] = 3
    
    # Create output directories
    log_dir = Path(config['output']['log_dir'])
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    plot_dir = Path(config['output']['plot_dir'])
    
    log_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    
    if args.hdf5:
        # Use HDF5 dataset if provided
        logger.info(f"Using HDF5 dataset: {args.hdf5}")
        
        # Create train and validation dataloaders directly
        train_loader, val_loader = create_train_val_dataloaders(
            hdf5_path=args.hdf5,
            batch_size=config['data']['batch_size'],
            augment=config['data']['augment'],
            augment_params=config.get('augment_params', {
                'scale_prob': 0.3,
                'noise_prob': 0.3,
                'shift_prob': 0.3,
            }),
            val_split=config['data']['train_val_test_split'][1] / (config['data']['train_val_test_split'][0] + config['data']['train_val_test_split'][1]),
            num_workers=config['data']['num_workers'],
            seed=args.seed
        )
        
        # Create a third dataloader for testing (using validation dataset)
        test_dataset = HDF5SegmentationDataset(
            hdf5_path=args.hdf5,
            augment=False,
            train=False,
            val_split=config['data']['train_val_test_split'][1] / (config['data']['train_val_test_split'][0] + config['data']['train_val_test_split'][1]),
            seed=args.seed
        )
        
        test_loader = test_dataset.get_dataloader(
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers']
        )
        
        logger.info(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} validation samples")
        
    else:
        # Use standard Fast5/TSV dataset
        fast5_dir = Path(config['data']['fast5_dir'])
        tsv_path = Path(config['data']['tsv_path'])
        
        fast5_reader = Fast5Reader(fast5_dir)
        label_reader = LabelReader(tsv_path)
        
        # Create dataset
        dataset = NanoporeSegmentationDataset(
            fast5_reader=fast5_reader,
            label_reader=label_reader,
            chunk_size=config['data']['chunk_size'],
            stride=config['data']['stride'],
            max_workers=config['data']['max_workers'],
            normalize=config['data']['normalize'],
            filter_signal=config['data']['filter_signal'],
            augment=config['data']['augment'],
            cache_size=config['data']['cache_size'],
        )
        
        # Split dataset
        split_ratios = config['data']['train_val_test_split']
        total_chunks = len(dataset)
        train_size = int(split_ratios[0] * total_chunks)
        val_size = int(split_ratios[1] * total_chunks)
        test_size = total_chunks - train_size - val_size
        
        logger.info(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test chunks")
        
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_chunks))
        
        # Limit data in debug mode
        if args.debug:
            train_indices = train_indices[:100]
            val_indices = val_indices[:20]
            test_indices = test_indices[:20]
        
        # Create subset samplers
        from torch.utils.data import SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config['data']['batch_size'],
            sampler=train_sampler,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            prefetch_factor=2
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config['data']['batch_size'],
            sampler=val_sampler,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            prefetch_factor=2
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config['data']['batch_size'],
            sampler=test_sampler,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            prefetch_factor=2
        )
    
    # Create model
    logger.info("Creating model...")
    model_type = config['model']['type']
    
    if model_type == 'SegmentationModel':
        model = SegmentationModel(
            input_channels=config['model']['input_channels'],
            hidden_channels=config['model']['hidden_channels'],
            num_residual_blocks=config['model']['num_residual_blocks'],
            dropout=config['model']['dropout'],
        )
    elif model_type == 'EfficientUNet':
        model = EfficientUNet(
            input_channels=config['model']['input_channels'],
            init_features=config['model']['init_features'],
            depth=config['model']['depth'],
            dropout=config['model']['dropout'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create loss function
    criterion = create_loss_function(config['training']['loss'])
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        config['training']['scheduler'],
        config['training']['num_epochs']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        eval_interval=config['training']['eval_interval'],
        save_best_only=config['training']['save_best_only'],
        early_stopping=config['training']['early_stopping'],
    )
    
    # Plot training curves
    logger.info("Plotting training curves...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if history['val_loss']:
        plot_loss_curves(
            history['train_loss'],
            history['val_loss'],
            output_path=plot_dir / f"loss_curves_{timestamp}.png"
        )
        
        if history['metrics']:
            metric_names = ['f1', 'precision', 'recall', 'f1_tol', 'precision_tol', 'recall_tol']
            plot_metric_curves(
                history['metrics'],
                metric_names,
                output_path=plot_dir / f"metric_curves_{timestamp}.png"
            )
    
    # Export to ONNX if specified
    if config['output']['export_onnx']:
        logger.info("Exporting model to ONNX...")
        onnx_output_path = Path(config['output']['onnx_output_path'])
        trainer.export_to_onnx(onnx_output_path)
    
    logger.info("Training completed.")

if __name__ == "__main__":
    main()