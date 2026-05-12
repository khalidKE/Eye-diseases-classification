"""
Evaluation Module for Eye Disease Classification

This module handles:
- Model evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- ROC curve plotting
- Per-class metrics
- Classification reports
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras

from src.config import CLASS_NAMES, CLASS_NAMES_DISPLAY

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def evaluate_model(
    model: keras.Model,
    test_data,
    class_names: List[str] = None,
    batch_size: int = 32
) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained Keras model
        test_data: Test data generator or dataset
        class_names: List of class names
        batch_size: Batch size for prediction
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    logger.info("Evaluating model...")
    
    # Keras 3 compatible: manually collect predictions and labels
    y_pred_prob = []
    y_true = []
    
    # Iterate through generator manually
    num_batches = 0
    max_batches = None
    
    if hasattr(test_data, 'samples') and hasattr(test_data, 'batch_size'):
        max_batches = (test_data.samples // test_data.batch_size) + 1
    
    for batch_x, batch_y in test_data:
        # Predict on batch
        batch_pred = model.predict(batch_x, verbose=0)
        y_pred_prob.append(batch_pred)
        
        # Get true labels
        if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
            # One-hot encoded
            batch_y_labels = np.argmax(batch_y, axis=1)
        else:
            # Already integer labels
            batch_y_labels = batch_y
        y_true.extend(batch_y_labels)
        
        num_batches += 1
        if max_batches and num_batches >= max_batches:
            break
        # Safety limit
        if num_batches > 1000:
            break
    
    # Concatenate all predictions
    y_pred_prob = np.concatenate(y_pred_prob, axis=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.array(y_true)
    
    logger.info(f"Evaluated on {len(y_true)} samples")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_prob, class_names)
    
    return metrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_prob: Predicted probabilities
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'overall': {},
        'per_class': {},
        'confusion_matrix': None
    }
    
    # Overall metrics
    metrics['overall']['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['overall']['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['overall']['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['overall']['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    metrics['overall']['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['overall']['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['overall']['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # ROC-AUC (One-vs-Rest)
    try:
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        roc_auc = roc_auc_score(y_true_bin, y_pred_prob, multi_class='ovr', average='macro')
        metrics['overall']['roc_auc_macro'] = float(roc_auc)
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        metrics['overall']['roc_auc_macro'] = None
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(np.sum(y_true == i))
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def print_metrics_report(metrics: Dict, class_names: List[str] = None):
    """
    Print a formatted metrics report.
    
    Args:
        metrics: Metrics dictionary from evaluate_model
        class_names: List of class names
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    
    # Overall metrics
    print("\nOverall Metrics:")
    print("-"*70)
    overall = metrics['overall']
    print(f"  Accuracy:           {overall['accuracy']:.4f}")
    print(f"  Precision (Macro):  {overall['precision_macro']:.4f}")
    print(f"  Recall (Macro):     {overall['recall_macro']:.4f}")
    print(f"  F1-Score (Macro):   {overall['f1_macro']:.4f}")
    print(f"  ROC-AUC (Macro):    {overall.get('roc_auc_macro', 'N/A')}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-"*70)
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*70)
    
    for class_name in class_names:
        class_metrics = metrics['per_class'].get(class_name, {})
        display_name = CLASS_NAMES_DISPLAY.get(class_name, class_name)
        print(f"{display_name:<25} "
              f"{class_metrics.get('precision', 0):>10.4f} "
              f"{class_metrics.get('recall', 0):>10.4f} "
              f"{class_metrics.get('f1_score', 0):>10.4f} "
              f"{class_metrics.get('support', 0):>10d}")
    
    print("="*70)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt if not normalize else '.2f',
        cmap='Blues',
        xticklabels=[CLASS_NAMES_DISPLAY.get(c, c) for c in class_names],
        yticklabels=[CLASS_NAMES_DISPLAY.get(c, c) for c in class_names]
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: List[str],
    save_path: str = None
):
    """
    Plot ROC curves for all classes.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
    """
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        display_name = CLASS_NAMES_DISPLAY.get(class_name, class_name)
        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f'{display_name} (AUC = {roc_auc:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: keras.callbacks.History,
    save_path: str = None
):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Keras history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_evaluation_report(
    metrics: Dict,
    model_name: str,
    save_dir: str = 'results'
):
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        model_name: Name of the model
        save_dir: Directory to save the report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    report_path = os.path.join(save_dir, f'{model_name}_evaluation.json')
    
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation report saved: {report_path}")


def compare_models(
    model_results: Dict[str, Dict],
    save_path: str = None
):
    """
    Compare multiple models and create comparison plot.
    
    Args:
        model_results: Dictionary mapping model names to their metrics
        save_path: Path to save comparison plot
    """
    model_names = list(model_results.keys())
    
    # Extract metrics
    accuracies = [model_results[m]['overall']['accuracy'] for m in model_names]
    precisions = [model_results[m]['overall']['precision_macro'] for m in model_names]
    recalls = [model_results[m]['overall']['recall_macro'] for m in model_names]
    f1s = [model_results[m]['overall']['f1_macro'] for m in model_names]
    
    # Create comparison plot
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3498db')
    ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#2ecc71')
    ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#e74c3c')
    ax.bar(x + 1.5*width, f1s, width, label='F1-Score', color='#f39c12')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage with dummy data
    logger.info("Testing evaluation module...")
    
    # Create dummy predictions for testing
    np.random.seed(42)
    y_true = np.random.randint(0, 4, size=100)
    y_pred = np.random.randint(0, 4, size=100)
    y_pred_prob = np.random.rand(100, 4)
    y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_prob, CLASS_NAMES)
    
    # Print report
    print_metrics_report(metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        save_path='results/test_confusion_matrix.png'
    )
    
    # Plot ROC curves
    plot_roc_curves(
        y_true, y_pred_prob, CLASS_NAMES,
        save_path='results/test_roc_curves.png'
    )
    
    logger.info("Evaluation module test complete!")
