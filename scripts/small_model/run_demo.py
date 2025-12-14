import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.small_model.best_model import TransformerEncoder, VectorRelationNet

def check_lfs_file(filepath):
    """Check if the file is a Git LFS pointer."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.read(100)
            if "version https://git-lfs.github.com/spec/v1" in header:
                return True
    except Exception:
        pass
    return False

def load_best_checkpoint(model_path, encoder, relation_net, device):
    # if check_lfs_file(model_path):
    #     print(f"Error: The file {model_path} is a Git LFS pointer, not the actual model weights.")
    #     print("Please run 'git lfs pull' to download the actual files.")
    #     sys.exit(1)
        
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Try to guess the keys
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            relation_net.load_state_dict(checkpoint['relation_net_state_dict'])
        elif 'model_state_dict' in checkpoint:
             # Maybe it's a single dict?
             pass
        else:
            # Fallback: assume the checkpoint IS the state dict or contains keys matching the models
            if 'encoder' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder'])
            if 'relation' in checkpoint:
                relation_net.load_state_dict(checkpoint['relation'])
            else:
                print("Warning: Could not identify model keys in checkpoint. Printing keys:")
                print(checkpoint.keys())
                
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"DEBUG: File size of {model_path} is {os.path.getsize(model_path)} bytes.")
        print("Tip: If the file size is small (<1KB), it is likely a placeholder text file, not the real model.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Best Lineage Demo")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the test data directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize Models
    encoder = TransformerEncoder().to(device)
    relation_net = VectorRelationNet(embedding_dim=128).to(device)
    
    # Load Best Checkpoint
    checkpoint_path = os.path.join(os.path.dirname(__file__), '../../data/models/small_model/best_lineage_model.pth')
    print(f"Loading best model from: {checkpoint_path}")
    load_best_checkpoint(checkpoint_path, encoder, relation_net, device)
    
    encoder.eval()
    relation_net.eval()
    
    # Load Data
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        sys.exit(1)
        
    files = [f for f in os.listdir(args.data_dir) if f.endswith('.pth')]
    if not files:
        print(f"No .pth files found in {args.data_dir}")
        sys.exit(1)
        
    print(f"Found {len(files)} data files. Starting evaluation...")
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for fname in files:
            fpath = os.path.join(args.data_dir, fname)
            try:
                data = torch.load(fpath, map_location=device)
                
                # Check keys
                if 'b_afea' not in data:
                    continue
                    
                b_afea = data['b_afea'].to(device)
                cfea = data['cfea'].to(device) # Child (Related)
                
                # Let's assume we evaluate positive pair first
                if b_afea.dim() == 2: b_afea = b_afea.unsqueeze(0)
                if cfea.dim() == 2: cfea = cfea.unsqueeze(0)
                
                vec1 = encoder(b_afea)
                vec2 = encoder(cfea)
                
                # What is the score?
                # If VectorRelationNet is used:
                out = relation_net(vec1, vec2)
                # Heuristic: use norm or mean as score?
                score = out.norm(dim=1).item() 
                
                all_scores.append(score)
                all_labels.append(1) # Positive
                
                # If negative data exists
                if 'fb_afea' in data and 'fcfea' in data:
                    fb_afea = data['fb_afea'].to(device)
                    fcfea = data['fcfea'].to(device)
                    if fb_afea.dim() == 2: fb_afea = fb_afea.unsqueeze(0)
                    if fcfea.dim() == 2: fcfea = fcfea.unsqueeze(0)
                    
                    vec1_neg = encoder(fb_afea)
                    vec2_neg = encoder(fcfea)
                    out_neg = relation_net(vec1_neg, vec2_neg)
                    score_neg = out_neg.norm(dim=1).item()
                    
                    all_scores.append(score_neg)
                    all_labels.append(0) # Negative
                    
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue

    if not all_scores:
        print("No scores computed.")
        sys.exit(0)
        
    # Calculate Metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_scores)
    print(f"AUC: {auc:.4f}")
    
    # TPR @ fixed FPR
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    def get_tpr_at_fpr(fpr_arr, tpr_arr, target):
        valid_indices = np.where(fpr_arr <= target)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            return tpr_arr[idx], fpr_arr[idx]
        else:
            return 0.0, 0.0

    tpr_1, fpr_1 = get_tpr_at_fpr(fpr, tpr, 0.01)
    tpr_01, fpr_01 = get_tpr_at_fpr(fpr, tpr, 0.001)
    
    print(f"TPR @ 1% FPR: {tpr_1:.4f} (Actual FPR: {fpr_1:.4f})")
    print(f"TPR @ 0.1% FPR: {tpr_01:.4f} (Actual FPR: {fpr_01:.4f})")

if __name__ == "__main__":
    main()
