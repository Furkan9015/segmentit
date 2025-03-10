"""Metrics for evaluating nanopore segmentation models."""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from pathlib import Path


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    distance_tolerance: int = 5
) -> Dict[str, float]:
    """Compute evaluation metrics for segmentation boundary detection.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        y_score: Predicted probability scores (optional)
        distance_tolerance: Number of positions to consider as correct for boundary detection
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Tolerance-based metrics (allowing boundaries to be off by a few positions)
    if distance_tolerance > 0:
        precision_tol, recall_tol, f1_tol = compute_tolerance_metrics(
            y_true, y_pred, tolerance=distance_tolerance
        )
        metrics['precision_tol'] = precision_tol
        metrics['recall_tol'] = recall_tol
        metrics['f1_tol'] = f1_tol
    
    # AUC metrics if probability scores are provided
    if y_score is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_score)
            
            # Precision-recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            metrics['pr_auc'] = auc(recall, precision)
        except ValueError:
            # Handle cases where there might be only one class
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    
    return metrics


def compute_tolerance_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 5
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score with position tolerance.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        tolerance: Number of positions to consider as correct
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if not np.any(y_true) or not np.any(y_pred):
        return 0.0, 0.0, 0.0
    
    # Find boundary positions
    true_positions = np.where(y_true == 1)[0]
    pred_positions = np.where(y_pred == 1)[0]
    
    if len(true_positions) == 0:
        return 0.0, 0.0, 0.0
    
    # For each predicted boundary, check if it's within tolerance of a true boundary
    true_positive = 0
    for pred_pos in pred_positions:
        distances = np.abs(true_positions - pred_pos)
        if np.min(distances) <= tolerance:
            true_positive += 1
    
    # Calculate metrics
    precision = true_positive / len(pred_positions) if len(pred_positions) > 0 else 0.0
    recall = true_positive / len(true_positions) if len(true_positions) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """Plot precision-recall curve.
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted probability scores
        output_path: Path to save the plot (optional)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """Plot ROC curve.
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted probability scores
        output_path: Path to save the plot (optional)
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    output_path: Optional[Path] = None
) -> None:
    """Plot training and validation loss curves.
    
    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metric_curves(
    metrics_list: List[Dict[str, float]],
    metric_names: List[str],
    output_path: Optional[Path] = None
) -> None:
    """Plot metrics over epochs.
    
    Args:
        metrics_list: List of metric dictionaries, one per epoch
        metric_names: List of metric names to plot
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(metrics_list) + 1)
    
    for metric_name in metric_names:
        if all(metric_name in metrics for metrics in metrics_list):
            values = [metrics[metric_name] for metrics in metrics_list]
            plt.plot(epochs, values, '-', label=metric_name)
    
    plt.title('Metrics over Training')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_signal_with_boundaries(
    signal: np.ndarray,
    true_boundaries: np.ndarray,
    pred_boundaries: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """Visualize signal with true and predicted boundaries.
    
    Args:
        signal: Signal values
        true_boundaries: Ground truth boundaries (binary)
        pred_boundaries: Predicted boundaries (binary)
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 8))
    
    # Plot signal
    plt.plot(signal, 'b-', alpha=0.7, label='Signal')
    
    # Plot true boundaries
    true_pos = np.where(true_boundaries == 1)[0]
    for pos in true_pos:
        plt.axvline(x=pos, color='g', linestyle='--', alpha=0.7)
    
    # Plot predicted boundaries
    pred_pos = np.where(pred_boundaries == 1)[0]
    for pos in pred_pos:
        plt.axvline(x=pos, color='r', linestyle=':', alpha=0.7)
    
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='b', lw=2),
        Line2D([0], [0], color='g', linestyle='--'),
        Line2D([0], [0], color='r', linestyle=':')
    ]
    plt.legend(custom_lines, ['Signal', 'True Boundaries', 'Predicted Boundaries'])
    
    plt.title('Signal with True and Predicted Boundaries')
    plt.xlabel('Position')
    plt.ylabel('Signal Value')
    plt.grid(True, alpha=0.3)
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()