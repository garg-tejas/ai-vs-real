import torch
import pandas as pd
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from utils.dataloader import get_test_loader
from models.convnext_model import build_convnext
from models.efficientnet_model import build_efficientnet
from models.resnet_model import build_resnet
from models.coatnet_model import build_coatnet
from models.hybrid_vit_model import build_hybrid_vit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2

def run_inference(model, test_csv, batch_size=64, data_dir='test_data_v2'):
    print(f"Running inference on {test_csv}")
    test_loader = get_test_loader(test_csv, batch_size=batch_size, data_dir=data_dir)
    predictions = []
    confidences = []
    true_labels = []
    has_labels = False
    
    pbar = tqdm(total=len(test_loader), desc="Inference")
    
    start_time = time.time()
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            conf = torch.softmax(outputs, dim=1)[:, 1]  
            
            predictions.extend(preds.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
            
            if labels[0] != -1:
                has_labels = True
                true_labels.extend(labels.numpy())
            
            pbar.update(1)
    
    pbar.close()
    inference_time = time.time() - start_time
    
    df = pd.read_csv(test_csv)
    df["label"] = predictions
    df["confidence"] = confidences
    
    Path("submissions").mkdir(exist_ok=True)
    
    model_name = getattr(model, 'name', 'model')
    submission_file = f"submissions/{model_name}_submission.csv"
    
    submission_df = df[["id", "label"]]
    submission_df.to_csv(submission_file, index=False)
    
    print(f"\nSaved predictions to {submission_file}")
    print(f"Inference time: {inference_time:.2f}s for {len(df)} images")
    print(f"Average time per image: {inference_time/len(df)*1000:.2f}ms")
    
    class_dist = np.bincount(predictions, minlength=NUM_CLASSES)
    print(f"\nClass distribution:")
    print(f"Real images (0): {class_dist[0]} ({class_dist[0]/len(predictions)*100:.2f}%)")
    print(f"AI-generated (1): {class_dist[1]} ({class_dist[1]/len(predictions)*100:.2f}%)")
    
    
    return predictions, confidences, true_labels if has_labels else None, inference_time

def load_model(model_name, model_path=None, model_dir="checkpoints"):
    """Load a pre-trained model for inference"""
    if model_name == "resnet":
        model = build_resnet(NUM_CLASSES)
        model.name = "resnet"
    elif model_name == "efficientnet":
        model = build_efficientnet(NUM_CLASSES)
        model.name = "efficientnet"
    elif model_name == "convnext":
        model = build_convnext(NUM_CLASSES)
        model.name = "convnext"
    elif model_name == "coatnet":
        model = build_coatnet(NUM_CLASSES)
        model.name = "coatnet"
    elif model_name == "hybrid_vit":
        model = build_hybrid_vit(NUM_CLASSES)
        model.name = "hybrid_vit"
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    if model_path is None:
        for metric in ['acc', 'f1', 'auc', 'final']:
            path = f"{model_dir}/{model_name}_best_{metric}.pth"
            if Path(path).exists():
                model_path = path
                break
                
        if model_path is None:
            raise FileNotFoundError(f"No checkpoint found for {model_name} model in {model_dir}")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Loaded {model_name} model from {model_path}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument("--model", type=str, default="convnext",
                        choices=["resnet", "efficientnet", "convnext", "coatnet", "hybrid_vit", "all"],
                        help="Model to use for inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    parser.add_argument("--test_csv", type=str, default="test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--data_dir", type=str, default="test_data_v2",
                        help="Directory containing test images")
    parser.add_argument("--model_dir", type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file name (default: submissions/<model_name>_submission.csv)")
    
    args = parser.parse_args()
    
    if args.model == "all":
        models = ["resnet", "efficientnet", "convnext", "coatnet", "hybrid_vit"]
        checkpoints = ["best_acc", "best_f1", "best_auc", "final"]
        results = []
        all_predictions = {}
        
        for model_name in models:
            print(f"\n{'='*50}")
            print(f"Running inference with {model_name} model")
            print(f"{'='*50}\n")
            
            for checkpoint in checkpoints:
                print(f"\nRunning inference with {model_name} {checkpoint} checkpoint\n")
                try:
                    checkpoint_path = f"{args.model_dir}/{model_name}_{checkpoint}.pth"
                    if not Path(checkpoint_path).exists():
                        print(f"Warning: Checkpoint {checkpoint_path} not found, skipping")
                        continue
                        
                    model = load_model(model_name, model_path=checkpoint_path, model_dir=args.model_dir)
                    model.name = f"{model_name}_{checkpoint}"
                    
                    preds, confs, labels, inference_time = run_inference(
                        model, args.test_csv, args.batch_size, data_dir=''
                    )
                    
                    all_predictions[f"{model_name}_{checkpoint}"] = {
                        'predictions': preds,
                        'confidences': confs
                    }
                    
                    total_params = sum(p.numel() for p in model.parameters())
                    
                    results.append({
                        "model_name": model_name,
                        "checkpoint": checkpoint,
                        "inference_time": inference_time,
                        "total_params": total_params,
                        "images_per_second": len(preds) / inference_time
                    })
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    print(f"Skipping {model_name} {checkpoint} model")
        
        if len(results) > 0:
            compare_model_predictions(all_predictions)
            
            print("\n\nInference Performance Comparison:")
            print("-" * 100)
            print(f"{'Model':<15} {'Checkpoint':<10} {'Params':<12} {'Total Time':<15} {'Time/Image':<15} {'Images/sec':<10}")
            print("-" * 100)
            
            for r in results:
                num_images = len(pd.read_csv(args.test_csv))
                print(f"{r['model_name']:<15} {r['checkpoint']:<10} {r['total_params']:,} {r['inference_time']:.2f}s " + 
                      f"{r['inference_time']/num_images*1000:.2f}ms {r['images_per_second']:.2f}")
    else:

        model = load_model(args.model, args.checkpoint, args.model_dir)
        run_inference(model, args.test_csv, args.batch_size, data_dir='')

def compare_model_predictions(predictions_dict):
    """Compare predictions across different models"""
    if len(predictions_dict) <= 1:
        return
    
    Path("plots").mkdir(exist_ok=True)
    
    model_names = list(predictions_dict.keys())
    first_model = model_names[0]
    num_samples = len(predictions_dict[first_model]['predictions'])
    
    agreement_matrix = np.zeros((len(model_names), len(model_names)))
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                agreement_matrix[i][j] = 1.0
                continue
                
            preds1 = predictions_dict[model1]['predictions']
            preds2 = predictions_dict[model2]['predictions']
            agreement = np.mean(preds1 == preds2)
            agreement_matrix[i][j] = agreement
    
    plt.figure(figsize=(10, 8))
    plt.imshow(agreement_matrix, cmap='viridis', vmin=0.5, vmax=1.0)
    plt.colorbar(label='Agreement Percentage')
    plt.title('Model Agreement Heatmap')
    plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
    plt.yticks(np.arange(len(model_names)), model_names)
    
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            plt.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if agreement_matrix[i, j] < 0.75 else "black")
    
    plt.tight_layout()
    plt.savefig("plots/model_agreement_heatmap.png")
    plt.close()
    
    disagreement_count = np.zeros(num_samples)
    
    for i in range(num_samples):
        votes = {}
        for model in model_names:
            pred = predictions_dict[model]['predictions'][i]
            votes[pred] = votes.get(pred, 0) + 1
        
        max_votes = max(votes.values())
        disagreement_count[i] = len(model_names) - max_votes

    most_contentious = np.argsort(disagreement_count)[::-1]
    num_contentious = min(10, np.sum(disagreement_count > 0))
    
    if num_contentious > 0:
        print(f"\nTop {num_contentious} most contentious predictions:")
        print("-" * 80)
        print(f"{'Image Index':<12} {'Predicted as Real':<25} {'Predicted as AI':<25}")
        print("-" * 80)
        
        for i in range(int(num_contentious)):
            idx = most_contentious[i]
            real_models = [model for model in model_names if predictions_dict[model]['predictions'][idx] == 0]
            ai_models = [model for model in model_names if predictions_dict[model]['predictions'][idx] == 1]
            
            print(f"{idx:<12} {', '.join(real_models):<25} {', '.join(ai_models):<25}")
    
    print(f"\nOverall model agreement: {np.mean(agreement_matrix - np.eye(len(model_names))):.4f}")
    
    ensemble_preds = []
    for i in range(num_samples):
        votes = {}
        for model in model_names:
            pred = predictions_dict[model]['predictions'][i]
            votes[pred] = votes.get(pred, 0) + 1
        
        max_vote_count = max(votes.values())
        winners = [k for k, v in votes.items() if v == max_vote_count]
        ensemble_pred = 1 if 1 in winners else 0
        ensemble_preds.append(ensemble_pred)
    
    df = pd.read_csv("test.csv")
    df["label"] = ensemble_preds
    
    Path("submissions").mkdir(exist_ok=True)
    
    submission_df = df[["id", "label"]]
    submission_df.to_csv("submissions/voting_ensemble_submission.csv", index=False)
    
    print(f"\nSaved voting ensemble predictions to submissions/voting_ensemble_submission.csv")

if __name__ == "__main__":
    main()