import argparse
import yaml
import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.small_model.dataset import get_dataloader
from src.small_model.model import get_model, save_model
from src.small_model.trainer import train_model
from src.small_model.utils import set_seed, get_device

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train Parent and Child Models")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # --- Train Parent Model ---
    print("\n=== Training Parent Model ===")
    parent_config = config['models']['parent']
    parent_classes = parent_config.get('classes')
    
    # Update num_classes based on selected classes if provided
    num_classes_parent = len(parent_classes) if parent_classes else config['dataset']['num_classes']
    
    train_loader_parent = get_dataloader(config, split='train', selected_classes=parent_classes)
    
    parent_model = get_model(
        parent_config['name'], 
        num_classes_parent, 
        pretrained=parent_config['pretrained']
    )
    
    parent_model = train_model(parent_model, train_loader_parent, parent_config, device)
    
    os.makedirs(os.path.dirname(parent_config['save_path']), exist_ok=True)
    save_model(parent_model, parent_config['save_path'])
    print(f"Parent model saved to {parent_config['save_path']}")

    # --- Train Child Model ---
    print("\n=== Training Child Model ===")
    child_config = config['models']['child']
    child_classes = child_config.get('classes')
    num_classes_child = len(child_classes) if child_classes else config['dataset']['num_classes']
    
    train_loader_child = get_dataloader(config, split='train', selected_classes=child_classes)
    
    child_model = get_model(
        child_config['name'], 
        num_classes_child, 
        pretrained=child_config['pretrained']
    )

    # Finetune from parent if specified
    if child_config.get('finetune_from') == 'parent':
        print("Loading parent weights for child initialization...")
        state_dict = torch.load(parent_config['save_path'], map_location=device)
        
        # Handle dimension mismatch if classes differ
        if num_classes_parent != num_classes_child:
            print(f"Warning: Parent has {num_classes_parent} classes, Child has {num_classes_child}. Reinitializing classifier layer.")
            # Remove classifier weights from state_dict
            # MobileNetV2: classifier[1]
            # ResNet: fc
            keys_to_remove = [k for k in state_dict.keys() if 'classifier.1' in k or 'fc.' in k]
            for k in keys_to_remove:
                del state_dict[k]
                
        child_model.load_state_dict(state_dict, strict=False)
    
    child_model = train_model(child_model, train_loader_child, child_config, device)
    
    os.makedirs(os.path.dirname(child_config['save_path']), exist_ok=True)
    save_model(child_model, child_config['save_path'])
    print(f"Child model saved to {child_config['save_path']}")

    # --- Train Independent Model ---
    if 'independent' in config['models']:
        print("\n=== Training Independent Model ===")
        ind_config = config['models']['independent']
        ind_classes = ind_config.get('classes')
        num_classes_ind = len(ind_classes) if ind_classes else config['dataset']['num_classes']
        
        train_loader_ind = get_dataloader(config, split='train', selected_classes=ind_classes)
        
        ind_model = get_model(
            ind_config['name'], 
            num_classes_ind, 
            pretrained=ind_config['pretrained']
        )
        
        ind_model = train_model(ind_model, train_loader_ind, ind_config, device)
        
        os.makedirs(os.path.dirname(ind_config['save_path']), exist_ok=True)
        save_model(ind_model, ind_config['save_path'])
        print(f"Independent model saved to {ind_config['save_path']}")

if __name__ == "__main__":
    main()
