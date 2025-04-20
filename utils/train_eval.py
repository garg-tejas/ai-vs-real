import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from torch.amp import autocast, GradScaler

def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=True, empty_cache_freq=0):
    """Train the model for one epoch and return metrics"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler('cuda', enabled=use_amp)
    
    for i, (imgs, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Forward pass with automatic mixed precision
        with autocast('cuda', enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        # Backward pass and optimize with gradient scaling (for AMP)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Track statistics
        total_loss += loss.item()
        with torch.no_grad():
            preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

def evaluate(model, loader, criterion, device):
    """Evaluate the model and return metrics"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # For AUC calculation
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            # Track statistics
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability of class 1
            preds = outputs.argmax(1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_preds), 
        np.array(all_labels), 
        np.array(all_probs)
    )
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

def calculate_metrics(preds, labels, probs=None):
    """Calculate classification metrics"""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = (preds == labels).mean()
    
    # Only calculate these if we have samples from both classes
    unique_classes = np.unique(labels)
    if len(unique_classes) > 1:
        # F1 score
        metrics['f1'] = f1_score(labels, preds)
        
        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        
        # Precision and recall
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # AUC (if probabilities are provided)
        if probs is not None:
            metrics['auc'] = roc_auc_score(labels, probs)
    else:
        # Set default values if only one class is present
        metrics['f1'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        if probs is not None:
            metrics['auc'] = 0.5
    
    return metrics

def print_metrics(metrics, prefix=""):
    """Print metrics in a readable format"""
    print(f"{prefix} Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}", end="")
    
    if 'f1' in metrics:
        print(f", F1: {metrics['f1']:.4f}", end="")
    
    if 'auc' in metrics:
        print(f", AUC: {metrics['auc']:.4f}", end="")
        
    print(f", Precision: {metrics.get('precision', 0):.4f}, Recall: {metrics.get('recall', 0):.4f}")