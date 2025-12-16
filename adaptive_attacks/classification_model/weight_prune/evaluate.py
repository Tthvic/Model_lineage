"""
Model Evaluation Script

Evaluate trained models on Caltech-101 dataset and generate performance reports.
"""

import os
import torch
import argparse
import json
from torch.utils.data import DataLoader
import numpy as np

from train_caltech import get_model, evaluate_model
from caltech101 import get_data_loader, PARENT_CLASSES, CHILD_CLASSES
import config as cfg
from utils import set_all_seeds, format_time
import time


def load_model(model_path, model_name, num_classes, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        model_name (str): Model architecture name
        num_classes (int): Number of output classes
        device: Device to load model on
    
    Returns:
        torch.nn.Module: Loaded model
    """
    # Create model with dummy args
    class Args:
        def __init__(self, device):
            self.device = device
    
    args = Args(device)
    model, _, _ = get_model(args, model_name, num_classes)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def evaluate_single_model(model_id, model_dir, class_subset, model_type='parent'):
    """
    Evaluate a single model.
    
    Args:
        model_id (int): Model identifier
        model_dir (str): Directory containing model checkpoints
        class_subset (list): Classes this model was trained on
        model_type (str): 'parent' or 'child'
    
    Returns:
        dict: Evaluation results
    """
    model_path = os.path.join(model_dir, f'{cfg.MODEL_PREFIX}_{model_id}.pth')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return None
    
    # Determine number of classes
    num_classes = cfg.PARENT_NUM_CLASSES if model_type == 'parent' else cfg.CHILD_NUM_CLASSES
    
    # Load model
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, cfg.MODEL_NAME, num_classes, device)
    
    # Get data loader
    _, test_loader = get_data_loader(class_subset)
    
    # Evaluate
    start_time = time.time()
    accuracy = evaluate_model(model, test_loader, device)
    eval_time = time.time() - start_time
    
    results = {
        'model_id': model_id,
        'model_type': model_type,
        'classes': class_subset,
        'accuracy': accuracy,
        'eval_time': eval_time
    }
    
    return results


def evaluate_all_models(model_dir, model_type='parent'):
    """
    Evaluate all models of a specific type.
    
    Args:
        model_dir (str): Directory containing model checkpoints
        model_type (str): 'parent' or 'child'
    
    Returns:
        list: List of evaluation results
    """
    print(f"\nEvaluating {model_type} models from {model_dir}")
    print("=" * 70)
    
    # Get class subsets
    if model_type == 'parent':
        class_subsets = PARENT_CLASSES
    else:
        class_subsets = CHILD_CLASSES
    
    results = []
    
    for i, class_subset in enumerate(class_subsets):
        print(f"\nEvaluating {model_type} model {i}...")
        print(f"Classes: {class_subset}")
        
        result = evaluate_single_model(i, model_dir, class_subset, model_type)
        
        if result is not None:
            results.append(result)
            print(f"Accuracy: {result['accuracy']:.2f}%")
            print(f"Evaluation time: {format_time(result['eval_time'])}")
        else:
            print(f"Skipped model {i} (not found)")
    
    return results


def generate_report(results, output_path='evaluation_report.json'):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results (list): List of evaluation results
        output_path (str): Path to save report
    """
    if not results:
        print("No results to report")
        return
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in results]
    
    report = {
        'num_models': len(results),
        'model_type': results[0]['model_type'],
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'median_accuracy': np.median(accuracies),
        'individual_results': results
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model Type: {report['model_type']}")
    print(f"Number of Models: {report['num_models']}")
    print(f"Mean Accuracy: {report['mean_accuracy']:.2f}% ± {report['std_accuracy']:.2f}%")
    print(f"Min Accuracy: {report['min_accuracy']:.2f}%")
    print(f"Max Accuracy: {report['max_accuracy']:.2f}%")
    print(f"Median Accuracy: {report['median_accuracy']:.2f}%")
    print(f"\nReport saved to: {output_path}")
    print("=" * 70)


def compare_models(parent_results, child_results, output_path='comparison_report.json'):
    """
    Compare parent and child model performance.
    
    Args:
        parent_results (list): Parent model results
        child_results (list): Child model results
        output_path (str): Path to save comparison report
    """
    parent_acc = [r['accuracy'] for r in parent_results]
    child_acc = [r['accuracy'] for r in child_results]
    
    comparison = {
        'parent_models': {
            'count': len(parent_results),
            'mean_accuracy': np.mean(parent_acc),
            'std_accuracy': np.std(parent_acc)
        },
        'child_models': {
            'count': len(child_results),
            'mean_accuracy': np.mean(child_acc),
            'std_accuracy': np.std(child_acc)
        },
        'accuracy_difference': np.mean(parent_acc) - np.mean(child_acc)
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"Parent Models: {comparison['parent_models']['mean_accuracy']:.2f}% "
          f"± {comparison['parent_models']['std_accuracy']:.2f}%")
    print(f"Child Models: {comparison['child_models']['mean_accuracy']:.2f}% "
          f"± {comparison['child_models']['std_accuracy']:.2f}%")
    print(f"Difference: {comparison['accuracy_difference']:.2f}%")
    print(f"\nComparison saved to: {output_path}")
    print("=" * 70)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_dir', type=str, default=cfg.PARENT_MODEL_DIR,
                       help='Directory containing model checkpoints')
    parser.add_argument('--model_type', type=str, default='parent',
                       choices=['parent', 'child', 'both'],
                       help='Type of models to evaluate')
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                       help='Output file for evaluation report')
    parser.add_argument('--seed', type=int, default=cfg.SEED,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_all_seeds(args.seed)
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Model Architecture: {cfg.MODEL_NAME}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Evaluate models
    if args.model_type == 'parent':
        results = evaluate_all_models(cfg.PARENT_MODEL_DIR, 'parent')
        generate_report(results, args.output)
    
    elif args.model_type == 'child':
        results = evaluate_all_models(cfg.CHILD_MODEL_DIR, 'child')
        generate_report(results, args.output)
    
    elif args.model_type == 'both':
        parent_results = evaluate_all_models(cfg.PARENT_MODEL_DIR, 'parent')
        child_results = evaluate_all_models(cfg.CHILD_MODEL_DIR, 'child')
        
        generate_report(parent_results, 'parent_evaluation.json')
        generate_report(child_results, 'child_evaluation.json')
        
        if parent_results and child_results:
            compare_models(parent_results, child_results)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
