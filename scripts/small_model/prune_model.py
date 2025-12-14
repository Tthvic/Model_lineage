import argparse
import yaml
import os
import sys
import torch
import torch.nn.utils.prune as prune

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.small_model.model import get_model, save_model
from src.small_model.utils import set_seed, get_device

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prune_model(model, amount=0.2, method='l1_unstructured'):
    print(f"Pruning model with amount={amount}, method={method}...")
    
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    if method == 'l1_unstructured':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    else:
        raise ValueError(f"Pruning method {method} not supported yet.")
    
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Prune a trained model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, default='child', choices=['parent', 'child', 'independent'], help='Which model to prune')
    parser.add_argument('--amount', type=float, default=0.2, help='Pruning amount (0.0-1.0)')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)
    device = get_device()

    # Load Model
    model_config = config['models'][args.model_type]
    model_path = model_config['save_path']
    classes = model_config.get('classes')
    num_classes = len(classes) if classes else config['dataset']['num_classes']
    
    print(f"Loading {args.model_type} model from {model_path}")
    model = get_model(model_config['name'], num_classes, pretrained=False, path=model_path)
    model = model.to(device)

    # Prune
    model = prune_model(model, amount=args.amount)
    
    # Save
    save_dir = os.path.dirname(model_path)
    save_name = os.path.basename(model_path).replace('.pth', f'_pruned_{args.amount}.pth')
    save_path = os.path.join(save_dir, save_name)
    
    save_model(model, save_path)
    print(f"Pruned model saved to {save_path}")

if __name__ == "__main__":
    main()
