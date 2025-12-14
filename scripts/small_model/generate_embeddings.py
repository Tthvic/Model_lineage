import argparse
import yaml
import os
import sys
import torch
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.small_model.dataset import get_dataloader
from src.small_model.model import get_model
from src.small_model.attack import MinADAttack
from src.small_model.utils import set_seed, get_device

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_target_samples(model, loader, num_classes, device):
    tgt_list = [None] * num_classes
    found_count = 0
    
    print("Finding target samples for each class...")
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in loader:
            if found_count == num_classes:
                break
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                
                if tgt_list[label] is None and label == pred:
                    tgt_list[label] = inputs[i].unsqueeze(0).clone()
                    found_count += 1
    
    for i in range(num_classes):
        if tgt_list[i] is None:
            tgt_list[i] = torch.zeros((1, 3, 224, 224)).to(device)
            
    return tgt_list

def main():
    parser = argparse.ArgumentParser(description="Generate MinAD Embeddings")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--split', type=str, default='train', help='Data split to use (train/test)')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)
    device = get_device()

    # Models to process
    models_to_process = ['child']
    if 'independent' in config['models']:
        models_to_process.append('independent')

    for model_key in models_to_process:
        print(f"\n=== Processing {model_key} model ===")
        model_config = config['models'][model_key]
        model_path = model_config['save_path']
        
        # Determine correct number of classes and specific classes list
        model_classes = model_config.get('classes')
        num_classes = len(model_classes) if model_classes else config['dataset']['num_classes']
        
        model_name = model_config['name']
        
        print(f"Loading model from {model_path} (num_classes={num_classes})")
        try:
            model = get_model(model_name, num_classes, pretrained=False, path=model_path)
        except FileNotFoundError:
            print(f"Model file not found: {model_path}. Skipping.")
            continue
            
        model = model.to(device)
        model.eval()

        # Load Data (Specific to this model's classes)
        loader = get_dataloader(config, split=args.split, selected_classes=model_classes)

        # Attack Setup
        attacker = MinADAttack(model, config, device)
        
        # Get Target Samples
        tgt_list = get_target_samples(model, loader, num_classes, device)

        # Parameters
        num_samples = config['embeddings'].get('num_samples', 100)
        num_target_classes = config['embeddings'].get('num_target_classes', 10)
        
        # Adjust num_target_classes if we have fewer classes than requested
        if num_target_classes >= num_classes:
            print(f"Warning: num_target_classes ({num_target_classes}) >= num_classes ({num_classes}). Adjusting to {num_classes - 1}.")
            num_target_classes = num_classes - 1
            
        if num_target_classes <= 0:
             print("Error: num_target_classes <= 0. Cannot generate embeddings.")
             continue
        
        embeddings = []
        labels_list = []
        
        count = 0
        pbar = tqdm(total=num_samples, desc=f"Generating Embeddings ({model_key})")
        
        for inputs, labels in loader:
            if count >= num_samples:
                break
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                top_preds = torch.argsort(outputs, dim=1, descending=True)
            
            for i in range(len(labels)):
                if count >= num_samples:
                    break
                    
                if preds[i] != labels[i]:
                    continue
                    
                x_ori = inputs[i].unsqueeze(0)
                y_ori = labels[i].unsqueeze(0)
                
                sample_distances = []
                
                target_indices = top_preds[i, 1:num_target_classes+1]
                
                for tgt_idx in target_indices:
                    tgt_class = tgt_idx.item()
                    target_sample = tgt_list[tgt_class]
                    adv_label = torch.tensor([tgt_class], device=device)
                    
                    try:
                        _, distance = attacker.attack(x_ori, y_ori, target_sample, adv_label)
                        sample_distances.append(distance)
                    except Exception as e:
                        sample_distances.append(0.0)
                
                if len(sample_distances) == num_target_classes:
                    embeddings.append(sample_distances)
                    labels_list.append(y_ori.item())
                    count += 1
                    pbar.update(1)

        pbar.close()
        
        # Save Results
        output_dir = config['embeddings']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_tensor = torch.tensor(embeddings)
        labels_tensor = torch.tensor(labels_list)
        
        save_name = f"{model_key}_{args.split}_embeddings.pt"
        torch.save({'embeddings': embeddings_tensor, 'labels': labels_tensor}, os.path.join(output_dir, save_name))
        print(f"Saved {model_key} embeddings to {os.path.join(output_dir, save_name)}")

if __name__ == "__main__":
    main()
