"""
Results Aggregation Script

This script aggregates pruning experiment results across multiple models and architectures.
It calculates weighted averages for:
- Rebuttal experiment: ResNet-50 and MobileNet (50% each)
- Revised experiment: Mixed tasks weighted by model counts
"""

import json
import os
import argparse
import numpy as np
from datetime import datetime


def load_json_results(filepath):
    """Load JSON results file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_pruning_results(results_list, weights=None):
    """
    Aggregate pruning results from multiple experiments.
    
    Args:
        results_list: List of result dictionaries
        weights: List of weights for each result (default: equal weights)
    
    Returns:
        dict: Aggregated results
    """
    if not results_list:
        return {}
    
    if weights is None:
        weights = [1.0 / len(results_list)] * len(results_list)
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # Initialize aggregated results
    aggregated = {
        'unpruned': 0.0,
        'pruned': {}
    }
    
    # Aggregate unpruned accuracy
    for result, weight in zip(results_list, weights):
        aggregated['unpruned'] += result.get('unpruned', 0) * weight
    
    # Aggregate pruned accuracies
    pruning_rates = set()
    for result in results_list:
        if 'pruned' in result:
            pruning_rates.update(result['pruned'].keys())
    
    for rate in pruning_rates:
        aggregated['pruned'][rate] = 0.0
        for result, weight in zip(results_list, weights):
            if 'pruned' in result and rate in result['pruned']:
                aggregated['pruned'][rate] += result['pruned'][rate] * weight
    
    return aggregated


def summarize_rebuttal_experiment(resnet_results, mobilenet_results):
    """
    Summarize Rebuttal experiment (6→6 task).
    
    Both ResNet-50 and MobileNet have 20 models each (50% weight each)
    """
    print("\n" + "="*60)
    print("REBUTTAL EXPERIMENT SUMMARY (6→6 Task)")
    print("="*60)
    
    # Equal weights for both architectures
    combined = aggregate_pruning_results(
        [resnet_results, mobilenet_results],
        weights=[0.5, 0.5]
    )
    
    print(f"\nResNet-50 Results ({20} models, 50% weight):")
    print(f"  Unpruned: {resnet_results['unpruned']:.2f}%")
    for rate in sorted(resnet_results.get('pruned', {}).keys()):
        print(f"  Pruned {rate}: {resnet_results['pruned'][rate]:.2f}%")
    
    print(f"\nMobileNet Results ({20} models, 50% weight):")
    print(f"  Unpruned: {mobilenet_results['unpruned']:.2f}%")
    for rate in sorted(mobilenet_results.get('pruned', {}).keys()):
        print(f"  Pruned {rate}: {mobilenet_results['pruned'][rate]:.2f}%")
    
    print(f"\nCombined Results (ResNet + MobileNet Average):")
    print(f"  Unpruned: {combined['unpruned']:.2f}%")
    for rate in sorted(combined['pruned'].keys()):
        print(f"  Pruned {rate}: {combined['pruned'][rate]:.2f}%")
    
    return combined


def summarize_revised_experiment(results_6cls, results_8cls, results_12cls):
    """
    Summarize Revised experiment (6→6/8/12 mixed tasks).
    
    Model distribution:
    - 6→6: 24 models (12 ResNet + 12 MobileNet) - 20% weight
    - 6→8: 36 models (18 ResNet + 18 MobileNet) - 30% weight
    - 6→12: 60 models (30 ResNet + 30 MobileNet) - 50% weight
    Total: 120 models
    """
    print("\n" + "="*60)
    print("REVISED EXPERIMENT SUMMARY (6→6/8/12 Mixed Tasks)")
    print("="*60)
    
    # Task weights based on model counts
    total_models = 24 + 36 + 60  # 120
    weights = [24/total_models, 36/total_models, 60/total_models]  # [0.2, 0.3, 0.5]
    
    # Aggregate across tasks
    combined = aggregate_pruning_results(
        [results_6cls, results_8cls, results_12cls],
        weights=weights
    )
    
    print(f"\n6→6 Task Results ({24} models, {weights[0]*100:.1f}% weight):")
    print(f"  Unpruned: {results_6cls['unpruned']:.2f}%")
    for rate in sorted(results_6cls.get('pruned', {}).keys()):
        print(f"  Pruned {rate}: {results_6cls['pruned'][rate]:.2f}%")
    
    print(f"\n6→8 Task Results ({36} models, {weights[1]*100:.1f}% weight):")
    print(f"  Unpruned: {results_8cls['unpruned']:.2f}%")
    for rate in sorted(results_8cls.get('pruned', {}).keys()):
        print(f"  Pruned {rate}: {results_8cls['pruned'][rate]:.2f}%")
    
    print(f"\n6→12 Task Results ({60} models, {weights[2]*100:.1f}% weight):")
    print(f"  Unpruned: {results_12cls['unpruned']:.2f}%")
    for rate in sorted(results_12cls.get('pruned', {}).keys()):
        print(f"  Pruned {rate}: {results_12cls['pruned'][rate]:.2f}%")
    
    print(f"\nWeighted Average (All Tasks Combined):")
    print(f"  Unpruned: {combined['unpruned']:.2f}%")
    for rate in sorted(combined['pruned'].keys()):
        print(f"  Pruned {rate}: {combined['pruned'][rate]:.2f}%")
    
    return combined


def extract_model_results(json_data, num_classes):
    """
    Extract results for a specific class configuration from JSON.
    
    Args:
        json_data: Loaded JSON results
        num_classes: Number of classes (6, 8, or 12)
    
    Returns:
        dict: Extracted results with unpruned and pruned accuracies
    """
    key = f'child_{num_classes}cls'
    if key not in json_data:
        return None
    
    # Aggregate across all models
    child_data = json_data[key]
    accuracies_unpruned = []
    accuracies_pruned = {}
    
    for model_id, model_data in child_data.items():
        if isinstance(model_data, dict) and 'results' in model_data:
            results = model_data['results']
            accuracies_unpruned.append(results.get('unpruned', 0))
            
            for rate, acc in results.get('pruned', {}).items():
                if rate not in accuracies_pruned:
                    accuracies_pruned[rate] = []
                accuracies_pruned[rate].append(acc)
    
    # Calculate averages
    avg_results = {
        'unpruned': np.mean(accuracies_unpruned) if accuracies_unpruned else 0,
        'pruned': {}
    }
    
    for rate, accs in accuracies_pruned.items():
        avg_results['pruned'][rate] = np.mean(accs) if accs else 0
    
    return avg_results


def main():
    parser = argparse.ArgumentParser(description='Summarize Pruning Experiment Results')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['rebuttal', 'revised'],
                       help='Experiment type to summarize')
    parser.add_argument('--resnet_results', type=str,
                       help='Path to ResNet-50 results JSON file')
    parser.add_argument('--mobilenet_results', type=str,
                       help='Path to MobileNet results JSON file')
    parser.add_argument('--results_6cls', type=str,
                       help='Path to 6-class results JSON (for revised)')
    parser.add_argument('--results_8cls', type=str,
                       help='Path to 8-class results JSON (for revised)')
    parser.add_argument('--results_12cls', type=str,
                       help='Path to 12-class results JSON (for revised)')
    parser.add_argument('--output', type=str,
                       help='Output file to save summary (optional)')
    
    args = parser.parse_args()
    
    if args.experiment == 'rebuttal':
        # Rebuttal experiment: 6→6 only
        if not args.resnet_results or not args.mobilenet_results:
            print("Error: --resnet_results and --mobilenet_results required for rebuttal")
            return
        
        resnet_data = load_json_results(args.resnet_results)
        mobilenet_data = load_json_results(args.mobilenet_results)
        
        # Extract 6-class results
        resnet_6cls = extract_model_results(resnet_data, 6)
        mobilenet_6cls = extract_model_results(mobilenet_data, 6)
        
        if not resnet_6cls or not mobilenet_6cls:
            print("Error: Could not find 6-class results in provided files")
            return
        
        summary = summarize_rebuttal_experiment(resnet_6cls, mobilenet_6cls)
        
    else:  # revised
        # Revised experiment: 6→6, 6→8, 6→12 mixed
        if not all([args.results_6cls, args.results_8cls, args.results_12cls]):
            print("Error: --results_6cls, --results_8cls, and --results_12cls required for revised")
            return
        
        data_6cls = load_json_results(args.results_6cls)
        data_8cls = load_json_results(args.results_8cls)
        data_12cls = load_json_results(args.results_12cls)
        
        # Extract results (each file should contain both ResNet and MobileNet combined)
        results_6 = extract_model_results(data_6cls, 6)
        results_8 = extract_model_results(data_8cls, 8)
        results_12 = extract_model_results(data_12cls, 12)
        
        if not all([results_6, results_8, results_12]):
            print("Error: Could not extract results from provided files")
            return
        
        summary = summarize_revised_experiment(results_6, results_8, results_12)
    
    # Save summary if output specified
    if args.output:
        output_data = {
            'experiment': args.experiment,
            'timestamp': datetime.now().isoformat(),
            'summary': summary
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSummary saved to: {args.output}")


if __name__ == '__main__':
    main()
