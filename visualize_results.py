import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_training_history(model_name):
    """Plot training history for a single model"""
    history_file = f"logs/{model_name}_history.json"
    
    if not os.path.exists(history_file):
        print(f"No history file found for {model_name}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10), dpi=300)
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if 'train_f1' in history and len(history['train_f1']) > 0:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['train_f1'], label='Train F1')
        plt.plot(epochs, history['val_f1'], label='Validation F1')
        plt.title(f'{model_name} - F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    if 'val_auc' in history and len(history['val_auc']) > 0:
        plt.subplot(2, 2, 4)
        plt.plot(epochs, history['val_auc'], label='Validation AUC')
        plt.title(f'{model_name} - AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path("plots").mkdir(exist_ok=True)
    
    plt.savefig(f"plots/{model_name}_training_curves.png")
    plt.close()
    
    print(f"Training plots for {model_name} saved to plots/{model_name}_training_curves.png")

def plot_comparison(comparison_file="logs/model_comparison.json"):
    """Plot comparison between different models"""
    if not os.path.exists(comparison_file):
        print(f"Comparison file {comparison_file} not found")
        return
    
    with open(comparison_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("No results to compare")
        return
    
    model_names = [r['model_name'] for r in results]
    accuracies = [r.get('best_val_acc', 0) for r in results]
    f1_scores = [r.get('best_val_f1', 0) for r in results]
    auc_scores = [r.get('best_val_auc', 0) for r in results]
    params = [r.get('total_params', 0) for r in results]
    train_times = [r.get('training_time', 0) / 60 for r in results]
    
    Path("plots").mkdir(exist_ok=True)
    
    plt.figure(figsize=(15, 8), dpi=300)
    x = np.arange(len(model_names))
    width = 0.2
    
    plt.bar(x - width, accuracies, width, label='Accuracy')
    plt.bar(x, f1_scores, width, label='F1 Score')
    plt.bar(x + width, auc_scores, width, label='AUC')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("plots/metrics_comparison.png")
    plt.close()
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(params, accuracies, s=100, alpha=0.7)
    
    for i, name in enumerate(model_names):
        plt.annotate(name, (params[i], accuracies[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Size vs. Performance')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("plots/params_vs_accuracy.png")
    plt.close()
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(train_times, accuracies, s=100, alpha=0.7)
    
    for i, name in enumerate(model_names):
        plt.annotate(name, (train_times[i], accuracies[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Training Time (minutes)')
    plt.ylabel('Validation Accuracy')
    plt.title('Training Time vs. Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("plots/time_vs_accuracy.png")
    plt.close()
    
    print(f"Comparison plots saved to plots/ directory")

def compare_submissions(submissions_dir="submissions"):
    """Compare different model submissions and analyze their differences"""
    if not os.path.exists(submissions_dir):
        print(f"Submissions directory {submissions_dir} not found")
        return
    
    submission_files = [f for f in Path(submissions_dir).glob("*.csv") if "acc" in f.name]
    
    if not submission_files:
        print("No submission files found")
        return
    
    submissions = {}
    
    for file_path in submission_files:
        model_name = file_path.stem.replace("_submission", "")
        df = pd.read_csv(file_path)
        
        if "id" not in df.columns or "label" not in df.columns:
            print(f"Skipping {file_path} as it doesn't have the expected columns")
            continue
        
        submissions[model_name] = df
    
    if len(submissions) < 2:
        print("Not enough submissions to compare")
        return
    
    model_names = list(submissions.keys())
    n_models = len(model_names)
    
    agreement_matrix = np.zeros((n_models, n_models))
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                agreement_matrix[i][j] = 1.0
                continue
            
            merged = pd.merge(submissions[model1], submissions[model2], 
                             on="id", suffixes=('_1', '_2'))
            agreement = (merged["label_1"] == merged["label_2"]).mean()
            agreement_matrix[i][j] = agreement
    
    plt.figure(figsize=(15, 12), dpi=300)
    plt.imshow(agreement_matrix, cmap='viridis', vmin=0.5, vmax=1.0)
    plt.colorbar(label='Agreement Percentage')
    plt.title('Model Predictions Agreement')
    plt.xticks(np.arange(n_models), model_names, rotation=45, ha="right")
    plt.yticks(np.arange(n_models), model_names)
    
    for i in range(n_models):
        for j in range(n_models):
            plt.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if agreement_matrix[i, j] < 0.75 else "black")
    
    plt.tight_layout()
    plt.savefig("plots/submission_agreement.png")
    plt.close()
    
    all_preds = {}
    
    for model in model_names:
        df = submissions[model]
        all_preds[model] = dict(zip(df["id"], df["label"]))
    
    contentious_images = []
    
    reference_model = model_names[0]
    for file_name in submissions[reference_model]["id"]:
        votes = {}
        for model in model_names:
            if file_name in all_preds[model]:
                pred = all_preds[model][file_name]
                votes[pred] = votes.get(pred, 0) + 1
        
        if len(votes) > 1:
            contentious_images.append({
                "file_name": file_name,
                "votes": votes,
                "max_vote_count": max(votes.values()),
                "disagreement": n_models - max(votes.values())
            })
    
    contentious_images.sort(key=lambda x: x["disagreement"], reverse=True)
    
    if contentious_images:
        top_n = min(20, len(contentious_images))
        top_contentious = contentious_images[:top_n]
        
        with open("logs/contentious_images.json", "w") as f:
            json.dump(top_contentious, f, indent=4)
        
        print(f"Found {len(contentious_images)} contentious images")
        print(f"Top {top_n} contentious images saved to logs/contentious_images.json")

    print("\nSubmission Agreement Summary:")
    overall_agreement = np.sum(agreement_matrix - np.eye(n_models)) / (n_models * (n_models - 1))
    print(f"Overall Agreement: {overall_agreement:.4f}")
    
    for i, model in enumerate(model_names):
        avg_agreement = np.sum(agreement_matrix[i, :] - agreement_matrix[i, i]) / (n_models - 1)
        print(f"{model}: Average agreement with other models = {avg_agreement:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Visualize AI vs Real model results")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to visualize (leave empty for all models)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare models using the model_comparison.json file")
    parser.add_argument("--submissions", action="store_true",
                        help="Compare different submission files")
    
    args = parser.parse_args()
    
    if not args.model and not args.compare and not args.submissions:
        args.compare = True
        args.submissions = True
        
        logs_dir = Path("logs")
        if logs_dir.exists():
            for file in logs_dir.glob("*_history.json"):
                model_name = file.stem.replace("_history", "")
                plot_training_history(model_name)
    else:
        if args.model:
            plot_training_history(args.model)
        
        if args.compare:
            plot_comparison()
        
        if args.submissions:
            compare_submissions()

if __name__ == "__main__":
    main()