"""
Pruning Experiment Script

This script runs the complete pruning experiment:
1. Train parent models (6-class)
2. For each child config (6, 8, 12 classes):
   - Fine-tune child models from parents
   - Evaluate unpruned child models
   - Prune child models at 20%, 30%, 50%, 70%
   - Evaluate pruned models
3. Log all results
"""

import os
import torch
import argparse
import json
from datetime import datetime
import copy

from train_caltech import get_model, evaluate_model, set_seed
from caltech101 import get_data_loader, PARENT_CLASSES
from tinyimagenet import (get_tinyimagenet_loader, CHILD_CLASSES_4, CHILD_CLASSES_6,
                          CHILD_CLASSES_6_REBUTTAL, CHILD_CLASSES_8, CHILD_CLASSES_12)
import config as cfg
from utils import setup_logger, prune_model, make_pruning_permanent, calculate_sparsity, log_metrics


def load_trained_model(model_path, model_name, num_classes, device):
    """Load a trained model from checkpoint"""
    class Args:
        def __init__(self, device):
            self.device = device
    
    args = Args(device)
    model, _, _ = get_model(args, model_name, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_parent_models(args, logger):
    """
    Evaluate all trained parent models.
    
    Returns:
        dict: Parent model results {model_id: accuracy}
    """
    # Get architecture-specific directory
    parent_dir = cfg.get_model_dir(args.model, 'parent')
    
    logger.info("=" * 60)
    logger.info(f"Evaluating Parent Models ({args.model}, 6-class)")
    logger.info("=" * 60)
    
    parent_results = {}
    
    for i, class_subset in enumerate(PARENT_CLASSES):
        model_path = os.path.join(parent_dir, f'{cfg.MODEL_PREFIX}_{i}.pth')
        
        if not os.path.exists(model_path):
            logger.warning(f"Parent model {i} not found at {model_path}")
            continue
        
        # Load model
        model = load_trained_model(model_path, args.model, cfg.PARENT_NUM_CLASSES, args.device)
        
        # Get data loader
        _, test_loader = get_data_loader(class_subset)
        
        # Evaluate
        accuracy = evaluate_model(model, test_loader, args.device)
        parent_results[i] = accuracy
        
        logger.info(f"Parent Model {i} ({args.model}): Accuracy = {accuracy:.2f}%")
    
    return parent_results
    
    return parent_results


def evaluate_child_with_pruning(model, test_loader, device, logger, model_id, num_classes, model_name, experiment, pruning_rates):
    """
    Evaluate a child model with different pruning rates.
    
    Args:
        pruning_rates: List of pruning rates to test
    
    Returns:
        dict: Results with pruning rates
    """
    results = {}
    
    # 1. Unpruned accuracy
    unpruned_acc = evaluate_model(model, test_loader, device)
    results['unpruned'] = unpruned_acc
    logger.info(f"  Unpruned: {unpruned_acc:.2f}%")
    
    # 2. Test different pruning rates
    for pruning_rate in pruning_rates:
        # Create a copy of the model for pruning
        pruned_model = copy.deepcopy(model)
        
        # Prune the model (manual magnitude-based pruning)
        pruned_model = prune_model(pruned_model, pruning_rate)
        
        # Calculate actual sparsity (after pruning)
        sparsity = calculate_sparsity(pruned_model)
        
        # Debug: Count pruned parameters directly from weights
        total_pruned = 0
        total_params = 0
        for name, module in pruned_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                weight = module.weight
                total_params += weight.numel()
                total_pruned += (weight == 0).sum().item()
        
        logger.info(f"  [DEBUG] Pruning {int(pruning_rate*100)}%: {total_pruned}/{total_params} params zeroed ({100.0*total_pruned/total_params:.2f}%)")
        
        # Evaluate pruned model
        pruned_acc = evaluate_model(pruned_model, test_loader, device)
        
        results[f'pruned_{int(pruning_rate*100)}'] = {
            'accuracy': pruned_acc,
            'sparsity': sparsity
        }
        
        logger.info(f"  Pruned {int(pruning_rate*100)}%: {pruned_acc:.2f}% (Sparsity: {sparsity:.2f}%)")
        
        # Save pruned model (architecture-specific directory with experiment type)
        if experiment:
            save_dir = f'./Cmodels_{num_classes}cls_{model_name}_{experiment}_pruned'
        else:
            save_dir = f'./Cmodels_{num_classes}cls_{model_name}_pruned'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{cfg.MODEL_PREFIX}_{model_id}_pruned{int(pruning_rate*100)}.pth')
        
        # Save pruned model (weights already permanently pruned by manual method)
        torch.save(pruned_model.state_dict(), save_path)
    
    return results


def run_child_experiment(args, logger, num_child_classes, pruning_rates):
    """
    Run complete experiment for a specific child class configuration.
    
    Args:
        args: Arguments
        logger: Logger instance
        num_child_classes (int): Number of classes for child models (4, 8, or 12)
        pruning_rates: List of pruning rates to test
    
    Returns:
        dict: Experiment results
    """
    # Get architecture-specific directories
    parent_dir = cfg.get_model_dir(args.model, 'parent')
    child_dir = cfg.get_model_dir(args.model, 'child', num_classes=num_child_classes, experiment=args.experiment)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Child Model Experiment ({args.model}): 6-class → {num_child_classes}-class")
    logger.info("=" * 60)
    
    # Select appropriate child class list (Tiny-ImageNet classes)
    if num_child_classes == 4:
        child_classes_list = CHILD_CLASSES_4  # Testing only
    elif num_child_classes == 6:
        # Use rebuttal list if specified, otherwise use revised
        child_classes_list = CHILD_CLASSES_6_REBUTTAL if args.experiment == 'rebuttal' else CHILD_CLASSES_6
    elif num_child_classes == 8:
        child_classes_list = CHILD_CLASSES_8  # Revised experiment
    elif num_child_classes == 12:
        child_classes_list = CHILD_CLASSES_12  # Revised experiment
    else:
        raise ValueError(f"Unsupported child class number: {num_child_classes}")
    
    # First, scan which models actually exist
    import glob
    existing_models = glob.glob(os.path.join(child_dir, f'{cfg.MODEL_PREFIX}_*.pth'))
    # Filter out epoch checkpoints, only get final models
    existing_models = [f for f in existing_models if 'epoch' not in f]
    existing_ids = sorted([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_models])
    
    logger.info(f"Found {len(existing_ids)} trained child models: {existing_ids}")
    
    experiment_results = {}
    
    if not existing_ids:
        logger.warning(f"No trained child models found in {child_dir}")
        return experiment_results
    
    num_parents = len(PARENT_CLASSES)
    
    for i in existing_ids:
        # Get corresponding class subset
        if i >= len(child_classes_list):
            logger.warning(f"Model ID {i} out of range for child_classes_list (len={len(child_classes_list)})")
            continue
            
        class_subset = child_classes_list[i]
        
        # Determine parent model
        parent_id = i % num_parents
        parent_model_path = os.path.join(parent_dir, f'{cfg.MODEL_PREFIX}_{parent_id}.pth')
        
        # Child model path (already confirmed to exist)
        child_model_path = os.path.join(child_dir, f'{cfg.MODEL_PREFIX}_{i}.pth')
        
        logger.info(f"\nChild Model {i} (from Parent {parent_id}, {args.model}, {num_child_classes}-class)")
        logger.info(f"Classes: {class_subset}")
        
        # Load child model
        model = load_trained_model(child_model_path, args.model, num_child_classes, args.device)
        
        # Get Tiny-ImageNet data loader (cross-domain evaluation with task-specific sample limits)
        samples_per_class = cfg.SAMPLES_PER_CLASS_CONFIG.get(num_child_classes, cfg.MAX_SAMPLES_PER_CLASS)
        _, test_loader = get_tinyimagenet_loader(class_subset, max_samples_per_class=samples_per_class)
        
        # Evaluate with different pruning rates
        results = evaluate_child_with_pruning(model, test_loader, args.device, logger, i, num_child_classes, args.model, args.experiment, pruning_rates)
        
        experiment_results[i] = {
            'parent_id': parent_id,
            'classes': class_subset,
            'results': results
        }
    
    return experiment_results


def main():
    parser = argparse.ArgumentParser(description='Pruning Experiment')
    parser.add_argument('--device', type=str, default=cfg.DEVICE,
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=cfg.SEED,
                       help='Random seed')
    parser.add_argument('--model', type=str, nargs='+', default=['resnet18', 'mobilenet'],
                       choices=['resnet18', 'mobilenet'],
                       help='Model architecture(s) to evaluate (can specify multiple)')
    parser.add_argument('--eval_parent', action='store_true',
                       help='Evaluate parent models')
    parser.add_argument('--eval_child', type=int, nargs='+', default=[4, 8, 12],
                       help='Child class numbers to evaluate (e.g., 4 6 8 12)')
    parser.add_argument('--experiment', type=str, default='revised',
                       choices=['rebuttal', 'revised'],
                       help='Experiment type: rebuttal or revised')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Pruning rates for both experiments
    pruning_rates = [0.2, 0.3, 0.5, 0.7]  # Both: 20%, 30%, 50%, 70%
    
    # Convert single model to list for consistency
    if isinstance(args.model, str):
        model_list = [args.model]
    else:
        model_list = args.model
    
    # Setup logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(cfg.LOG_DIR, f'pruning_experiment_{timestamp}.log')
    logger = setup_logger('pruning_experiment', log_file)
    
    logger.info("=" * 60)
    logger.info("Hierarchical Fine-Tuning with Pruning Experiment")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Models: {', '.join(model_list)}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Pruning rates: {pruning_rates}")
    logger.info("=" * 60)
    
    # Store all results
    all_results = {
        'timestamp': timestamp,
        'config': {
            'models': model_list,
            'device': args.device,
            'experiment': args.experiment,
            'seed': args.seed,
            'pruning_rates': cfg.PRUNING_RATES
        }
    }
    
    # Evaluate each model architecture
    for model_arch in model_list:
        # Create a copy of args with single model
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_arch
        
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Evaluating: {model_arch.upper()}")
        logger.info(f"{'#' * 60}")
        
        # 1. Evaluate parent models
        if args.eval_parent:
            parent_results = evaluate_parent_models(model_args, logger)
            all_results[f'{model_arch}_parent_models'] = parent_results
        
        # 2. Run child experiments for different class configurations
        for num_classes in args.eval_child:
            if num_classes in [4, 6, 8, 12]:
                child_results = run_child_experiment(model_args, logger, num_classes, pruning_rates)
                all_results[f'{model_arch}_child_{num_classes}cls'] = child_results
    
    # 3. Save results
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(cfg.RESULTS_DIR, f'pruning_results_{args.experiment}_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'=' * 60}")
    
    # Print summary
    logger.info("\nEXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    # Summary for each architecture
    for model_arch in model_list:
        logger.info(f"\n[{model_arch.upper()}]")
        
        # Parent models
        parent_key = f'{model_arch}_parent_models'
        if parent_key in all_results:
            parent_accs = list(all_results[parent_key].values())
            logger.info(f"  Parent Models (6-class): {len(parent_accs)} models")
            logger.info(f"    Mean Accuracy: {sum(parent_accs)/len(parent_accs):.2f}%")
        
        # Child models
        for num_classes in args.eval_child:
            key = f'{model_arch}_child_{num_classes}cls'
            if key in all_results:
                logger.info(f"\n  Child Models ({num_classes}-class): {len(all_results[key])} models")
                
                # Calculate average accuracies
                unpruned_accs = []
                pruned_20_accs = []
                pruned_30_accs = []
                pruned_50_accs = []
                pruned_70_accs = []
                
                for model_results in all_results[key].values():
                    results = model_results['results']
                    unpruned_accs.append(results['unpruned'])
                    if 'pruned_20' in results:
                        pruned_20_accs.append(results['pruned_20']['accuracy'])
                    if 'pruned_30' in results:
                        pruned_30_accs.append(results['pruned_30']['accuracy'])
                    if 'pruned_50' in results:
                        pruned_50_accs.append(results['pruned_50']['accuracy'])
                    if 'pruned_70' in results:
                        pruned_70_accs.append(results['pruned_70']['accuracy'])
                
                if unpruned_accs:
                    logger.info(f"    Unpruned: {sum(unpruned_accs)/len(unpruned_accs):.2f}%")
                if pruned_20_accs:
                    logger.info(f"    Pruned 20%: {sum(pruned_20_accs)/len(pruned_20_accs):.2f}%")
                if pruned_30_accs:
                    logger.info(f"    Pruned 30%: {sum(pruned_30_accs)/len(pruned_30_accs):.2f}%")
                if pruned_50_accs:
                    logger.info(f"    Pruned 50%: {sum(pruned_50_accs)/len(pruned_50_accs):.2f}%")
                if pruned_70_accs:
                    logger.info(f"    Pruned 70%: {sum(pruned_70_accs)/len(pruned_70_accs):.2f}%")
    
    # Calculate weighted average across all architectures (if multiple)
    if len(model_list) > 1:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"WEIGHTED AVERAGE (50% {model_list[0].upper()} + 50% {model_list[1].upper()})")
        logger.info("=" * 60)
        
        for num_classes in args.eval_child:
            # Collect results from all architectures
            arch_results = {}
            for model_arch in model_list:
                key = f'{model_arch}_child_{num_classes}cls'
                if key in all_results:
                    arch_results[model_arch] = all_results[key]
            
            if len(arch_results) == 2:  # Both architectures present
                logger.info(f"\nChild Models ({num_classes}-class):")
                
                # Calculate per-architecture averages
                arch_averages = {}
                for model_arch, results_dict in arch_results.items():
                    unpruned = []
                    pruned_20 = []
                    pruned_30 = []
                    pruned_50 = []
                    pruned_70 = []
                    
                    for model_results in results_dict.values():
                        results = model_results['results']
                        unpruned.append(results['unpruned'])
                        if 'pruned_20' in results:
                            pruned_20.append(results['pruned_20']['accuracy'])
                        if 'pruned_30' in results:
                            pruned_30.append(results['pruned_30']['accuracy'])
                        if 'pruned_50' in results:
                            pruned_50.append(results['pruned_50']['accuracy'])
                        if 'pruned_70' in results:
                            pruned_70.append(results['pruned_70']['accuracy'])
                    
                    arch_averages[model_arch] = {
                        'unpruned': sum(unpruned)/len(unpruned) if unpruned else 0,
                        'pruned_20': sum(pruned_20)/len(pruned_20) if pruned_20 else 0,
                        'pruned_30': sum(pruned_30)/len(pruned_30) if pruned_30 else 0,
                        'pruned_50': sum(pruned_50)/len(pruned_50) if pruned_50 else 0,
                        'pruned_70': sum(pruned_70)/len(pruned_70) if pruned_70 else 0,
                    }
                
                # Calculate 50-50 weighted average
                weighted_avg = {}
                for metric in ['unpruned', 'pruned_20', 'pruned_30', 'pruned_50', 'pruned_70']:
                    values = [arch_averages[arch][metric] for arch in model_list if arch in arch_averages]
                    if values:
                        weighted_avg[metric] = sum(values) / len(values)  # Equal weight
                
                logger.info(f"  Unpruned: {weighted_avg.get('unpruned', 0):.2f}%")
                # Only show pruning rates that have actual data
                if weighted_avg.get('pruned_20', 0) > 0:
                    logger.info(f"  Pruned 20%: {weighted_avg.get('pruned_20', 0):.2f}%")
                if weighted_avg.get('pruned_30', 0) > 0:
                    logger.info(f"  Pruned 30%: {weighted_avg.get('pruned_30', 0):.2f}%")
                if weighted_avg.get('pruned_50', 0) > 0:
                    logger.info(f"  Pruned 50%: {weighted_avg.get('pruned_50', 0):.2f}%")
                if weighted_avg.get('pruned_70', 0) > 0:
                    logger.info(f"  Pruned 70%: {weighted_avg.get('pruned_70', 0):.2f}%")


if __name__ == '__main__':
    main()
