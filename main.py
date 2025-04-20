import torch
import torch.nn as nn
import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils.dataloader import get_loaders
from utils.train_eval import train_one_epoch, evaluate, print_metrics
from models.convnext_model import build_convnext
from models.efficientnet_model import build_efficientnet
from models.resnet_model import build_resnet
from models.coatnet_model import build_coatnet
from models.hybrid_vit_model import build_hybrid_vit

def train_model(model_name, num_epochs=10, batch_size=64, seed=42, data_dir='train_data', output_dir='checkpoints', 
             early_stopping=True, patience=5, monitor='val_loss'):
    """
    Train and evaluate a model
    
    Parameters:
    -----------
    model_name: str
        Name of the model to train
    num_epochs: int
        Maximum number of epochs to train
    batch_size: int
        Batch size for training
    seed: int
        Random seed for reproducibility
    data_dir: str
        Directory containing training images
    output_dir: str
        Directory to save model checkpoints
    early_stopping: bool
        Whether to use early stopping
    patience: int
        Number of epochs to wait after validation performance stops improving
    monitor: str
        Metric to monitor for early stopping ('val_loss', 'val_acc', 'val_f1', 'val_auc')
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 2
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    Path("logs").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    
    print(f"Training {model_name} model on {DEVICE}")
    print(f"Models will be saved to {output_dir}/")
    
    train_loader, val_loader = get_loaders("train.csv", batch_size=batch_size, data_dir=data_dir)
    
    if model_name == "resnet":
        model = build_resnet(NUM_CLASSES)
    elif model_name == "efficientnet":
        model = build_efficientnet(NUM_CLASSES)
    elif model_name == "convnext":
        model = build_convnext(NUM_CLASSES)
    elif model_name == "coatnet":
        model = build_coatnet(NUM_CLASSES)
    elif model_name == "hybrid_vit":
        model = build_hybrid_vit(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    training_history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
        "learning_rates": [],
        "epoch_time": []
    }
    
    early_stop_counter = 0
    best_monitor_value = float('inf') if monitor == 'val_loss' else 0.0
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print_metrics(train_metrics, prefix="Train:")
        
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        print_metrics(val_metrics, prefix="Val:  ")
        
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        training_history["train_loss"].append(train_metrics["loss"])
        training_history["train_acc"].append(train_metrics["accuracy"])
        training_history["train_f1"].append(train_metrics.get("f1", 0))
        training_history["val_loss"].append(val_metrics["loss"])
        training_history["val_acc"].append(val_metrics["accuracy"])
        training_history["val_f1"].append(val_metrics.get("f1", 0))
        training_history["val_auc"].append(val_metrics.get("auc", 0))
        training_history["learning_rates"].append(current_lr)
        training_history["epoch_time"].append(epoch_time)
        
        print(f"Time: {epoch_time:.2f}s, Learning rate: {current_lr:.1e}")
        
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), f"{output_dir}/{model_name}_best_acc.pth")
            print(f"New best accuracy: {best_val_acc:.4f}")
        
        if "f1" in val_metrics and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), f"{output_dir}/{model_name}_best_f1.pth")
            print(f"New best F1 score: {best_val_f1:.4f}")
        
        if "auc" in val_metrics and val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), f"{output_dir}/{model_name}_best_auc.pth")
            print(f"New best AUC: {best_val_auc:.4f}")
        
        if early_stopping:
            monitor_key = monitor.replace('val_', '')
            if monitor_key == 'acc':
                monitor_key = 'accuracy'
                
            current_monitor_value = val_metrics[monitor_key]
            if (monitor == 'val_loss' and current_monitor_value < best_monitor_value) or \
               (monitor != 'val_loss' and current_monitor_value > best_monitor_value):
                best_monitor_value = current_monitor_value
                early_stop_counter = 0
                print(f"Early stopping counter reset: best {monitor} = {best_monitor_value:.4f}")
            else:
                early_stop_counter += 1
                print(f"Early stopping counter: {early_stop_counter}/{patience}")
            
            if early_stop_counter >= patience:
                print(f"\nEarly stopping triggered. {monitor} didn't improve for {patience} epochs.")
                break
    
    torch.save(model.state_dict(), f"{output_dir}/{model_name}_final.pth")
    
    with open(f"logs/{model_name}_history.json", "w") as f:
        json.dump(training_history, f)
    
    plot_training_history(training_history, model_name)
    
    total_time = time.time() - start_time
    
    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "best_val_auc": best_val_auc,
        "final_val_acc": val_metrics["accuracy"],
        "final_train_acc": train_metrics["accuracy"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_time": total_time,
        "avg_epoch_time": sum(training_history["epoch_time"]) / len(training_history["epoch_time"]),
    }

def plot_training_history(history, model_name):
    """Plot accuracy and loss curves"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    if 'train_f1' in history and len(history['train_f1']) > 0:
        plt.plot(history['train_f1'], label='Train F1', linestyle='--')
    if 'val_f1' in history and len(history['val_f1']) > 0:
        plt.plot(history['val_f1'], label='Validation F1', linestyle='--')
    plt.title(f'{model_name} - Accuracy & F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_training_history.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train different models for AI vs Real image classification")
    parser.add_argument("--model", type=str, default="all", 
                        choices=["resnet", "efficientnet", "convnext", "coatnet", "hybrid_vit", "all"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="train_data", help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--early_stopping", action="store_true", default=True, help="Use early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--monitor", type=str, default="val_loss", 
                        choices=["val_loss", "val_acc", "val_f1", "val_auc"],
                        help="Metric to monitor for early stopping")
    
    args = parser.parse_args()
    
    if args.model == "all":
        models = ["resnet", "efficientnet", "convnext", "coatnet", "hybrid_vit"]
        results = []
        
        for model_name in models:
            print(f"\n{'='*50}")
            print(f"Training {model_name} model")
            print(f"{'='*50}\n")
            
            if torch.cuda.is_available():
                print("Clearing CUDA cache before training...")
                torch.cuda.empty_cache()
            
            try:
                model_results = train_model(
                    model_name, 
                    args.epochs, 
                    args.batch_size, 
                    args.seed, 
                    args.data_dir, 
                    args.output_dir,
                    args.early_stopping,
                    args.patience,
                    args.monitor
                )
                results.append(model_results)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        with open("logs/model_comparison.json", "w") as f:
            json.dump(results, f, indent=4)
        
        print("\n\nModel Comparison:")
        print("-" * 110)
        print(f"{'Model':<15} {'Acc':<8} {'F1':<8} {'AUC':<8} {'Params':<12} {'Train Time':<15} {'Epoch Time':<10}")
        print("-" * 110)
        
        for r in results:
            print(f"{r['model_name']:<15} {r['best_val_acc']:.4f} {r.get('best_val_f1', 0):.4f} {r.get('best_val_auc', 0):.4f} {r['total_params']:,} {r['training_time']/60:.2f} min {r['avg_epoch_time']:.2f}s")
        
        create_comparison_plots(results)
    else:
        train_model(
            args.model, 
            args.epochs, 
            args.batch_size, 
            args.seed, 
            args.data_dir, 
            args.output_dir,
            args.early_stopping,
            args.patience,
            args.monitor
        )

def create_comparison_plots(results):
    """Create comparison plots between models"""
    model_names = [r['model_name'] for r in results]
    accuracies = [r['best_val_acc'] for r in results]
    f1_scores = [r.get('best_val_f1', 0) for r in results]
    auc_scores = [r.get('best_val_auc', 0) for r in results]
    
    cnn_models = ['resnet', 'efficientnet', 'convnext']
    hybrid_models = ['coatnet', 'hybrid_vit']
    
    cnn_indices = [i for i, name in enumerate(model_names) if name in cnn_models]
    hybrid_indices = [i for i, name in enumerate(model_names) if name in hybrid_models]
    
    cnn_acc = [accuracies[i] for i in cnn_indices]
    hybrid_acc = [accuracies[i] for i in hybrid_indices]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(model_names, accuracies, color=['blue' if name in cnn_models else 'green' for name in model_names])
    plt.title('Accuracy by Model')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', 
                ha='center', va='bottom', rotation=0)
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(model_names, f1_scores, color=['blue' if name in cnn_models else 'green' for name in model_names])
    plt.title('F1 Score by Model')
    plt.ylabel('Validation F1 Score')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', 
                ha='center', va='bottom', rotation=0)
    
    plt.subplot(2, 2, 3)
    bars = plt.bar(model_names, auc_scores, color=['blue' if name in cnn_models else 'green' for name in model_names])
    plt.title('AUC by Model')
    plt.ylabel('Validation AUC')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', 
                ha='center', va='bottom', rotation=0)
    
    plt.subplot(2, 2, 4)
    labels = ['CNN Models', 'Hybrid Models']
    avg_acc = [sum(cnn_acc)/len(cnn_acc), sum(hybrid_acc)/len(hybrid_acc)]
    bars = plt.bar(labels, avg_acc, color=['blue', 'green'])
    plt.title('Average Accuracy: CNN vs Hybrid')
    plt.ylabel('Average Validation Accuracy')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', 
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    param_counts = [r['total_params'] for r in results]
    
    plt.scatter(param_counts, accuracies, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        plt.annotate(name, (param_counts[i], accuracies[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Size vs. Performance')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/params_vs_accuracy.png")
    plt.close()

if __name__ == "__main__":
    main()