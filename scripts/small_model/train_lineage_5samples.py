import argparse
import yaml
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.small_model.utils import set_seed, get_device
from src.small_model.models import TransformerEncoder, VectorRelationNet, SimpleMLPEncoder

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class LineageDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(LineageDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def evaluate_pretrained_model(checkpoint_path, data_dir, device, model_arch='mobilenet'):
    """
    Evaluates a pre-trained Transformer+RelationNet model on the provided data directory.
    """
    print(f"\nEvaluating Pre-trained Model from: {checkpoint_path}")
    print(f"Test Data Directory: {data_dir}")
    print(f"Model Architecture: {model_arch}")
    
    # Determine feature dimension based on architecture
    if model_arch == 'mobilenet':
        feat_dim = 1280
        # Initialize Models
        encoder = TransformerEncoder(feat_dim=feat_dim).to(device)
    elif model_arch == 'resnet':
        feat_dim = 512
        # Initialize Models (ResNet uses SimpleMLPEncoder despite legacy naming)
        encoder = SimpleMLPEncoder(feat_dim=feat_dim).to(device)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    relation_net = VectorRelationNet(embedding_dim=128).to(device)
    
    # Load Checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'encnet_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['encnet_state_dict'])
                relation_net.load_state_dict(checkpoint['prenet_state_dict'])
            elif 'encoder_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                relation_net.load_state_dict(checkpoint['relation_net_state_dict'])
            else:
                # Fallback: try to load directly if keys match
                try:
                    encoder.load_state_dict(checkpoint['encoder'])
                    relation_net.load_state_dict(checkpoint['relation'])
                except:
                    print("Warning: Could not identify model keys. Attempting direct load...")
    except Exception as e:
        print(f"\n[ERROR] Failed to load checkpoint: {e}")
        print(f"File: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            print(f"Size: {os.path.getsize(checkpoint_path)} bytes")
            if os.path.getsize(checkpoint_path) < 1024:
                 print("\n[DIAGNOSIS] The file is too small (<1KB). It is likely a Git LFS pointer file.")
                 print("To fix this: Please upload the actual binary model file (usually >1MB) to this location.")
        else:
            print("\n[DIAGNOSIS] The checkpoint file does not exist at the specified path.")
        return

    encoder.eval()
    relation_net.eval()
    
    all_scores = []
    all_labels = []

    # Handle bundled data file (legacy format)
    if os.path.isfile(data_dir):
        print(f"Loading bundled data from {data_dir}...")
        try:
            data_bundle = torch.load(data_dir, map_location=device)
            
            # Map categories to labels
            # parent_child (formerly fuzi) -> 1
            # grandparent_child (formerly yesun) -> 1  
            # non (Negative) -> 0
            
            # Category name mapping for display
            category_display_names = {
                'fuzi': 'parent_child',
                'yesun': 'grandparent_child',
                'non': 'non_related'
            }
            
            categories = data_bundle.keys()
            print(f"Found categories: {[category_display_names.get(cat, cat) for cat in categories]}")
            
            with torch.no_grad():
                for category, samples in data_bundle.items():
                    label = 0 if 'non' in category.lower() else 1
                    display_name = category_display_names.get(category, category)
                    print(f"Processing {display_name} ({len(samples)} samples) as Label {label}")
                    
                    for i, sample in enumerate(samples):
                        try:
                            # Extract features
                            # Keys might vary slightly
                            if 'b_afea' in sample:
                                b_afea = sample['b_afea'].to(device)
                                cfea = sample['cfea'].to(device)
                                pfea = sample['pfea'].to(device)
                                # print(b_afea.shape,cfea.shape,"shaperrr")
                            else:
                                continue

                            # Ensure dimensions (Batch size 1)
                            # Input shape from file is [seq_len, feat_dim] e.g. [8, 1280]
                            # Model expects [batch, seq_len, feat_dim]
                            
                            if b_afea.dim() == 2: 
                                b_afea = b_afea.unsqueeze(0)
                            elif b_afea.dim() == 1:
                                # If it's 1D, it might be [feat_dim], so we need [1, 1, feat_dim]
                                b_afea = b_afea.unsqueeze(0).unsqueeze(0)
                                
                            if cfea.dim() == 2: 
                                cfea = cfea.unsqueeze(0)
                            elif cfea.dim() == 1:
                                cfea = cfea.unsqueeze(0).unsqueeze(0)
                                
                            if pfea.dim() == 2: 
                                pfea = pfea.unsqueeze(0)
                            elif pfea.dim() == 1:
                                pfea = pfea.unsqueeze(0).unsqueeze(0)

                            
                            # Forward Pass (Legacy Logic)
                            # 1. Encode Child and Ancestor(s)
                            enccfea = encoder(cfea)
                            enb_apfea = encoder(b_afea)
                            
                            # 2. Encode Parent (Target)
                            pknow = encoder(pfea)
                            
                            # 3. Predict Relation Vector
                            fsumknow = relation_net(enb_apfea, enccfea)
                            
                            # 4. Calculate Cosine Similarity
                            sim = torch.nn.functional.cosine_similarity(pknow, fsumknow).item()
                            
                            all_scores.append(sim)
                            all_labels.append(label)
                            
                        except Exception as e:
                            if i == 0: print(f"Error processing sample in {category}: {e}")
                            continue
                            
        except Exception as e:
            print(f"Error loading data bundle: {e}")
            return

    elif os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.pth')]
        if not files:
            print(f"No .pth files found in {data_dir}")
            return
            
        print(f"Found {len(files)} test files.")
        
        with torch.no_grad():
            for fname in files:
                fpath = os.path.join(data_dir, fname)
                try:
                    data = torch.load(fpath, map_location=device)
                    
                    # Check keys
                    if 'b_afea' not in data:
                        continue
                        
                    b_afea = data['b_afea'].to(device)
                    cfea = data['cfea'].to(device) # Child (Related)
                    
                    # Ensure dimensions
                    if b_afea.dim() == 2: b_afea = b_afea.unsqueeze(0)
                    if cfea.dim() == 2: cfea = cfea.unsqueeze(0)
                    
                    # Forward Pass
                    vec1 = encoder(b_afea)
                    vec2 = encoder(cfea)
                    out = relation_net(vec1, vec2)
                    
                    # Score: Norm of the output vector (heuristic from legacy code usage)
                    score = out.norm(dim=1).item() 
                    
                    all_scores.append(score)
                    all_labels.append(1) # Positive
                    
                    # Negative Pair
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
    else:
        print(f"Error: {data_dir} is not a valid file or directory.")
        return

    if not all_scores:
        print("No scores computed.")
        return

    # Metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_scores)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    def get_tpr_at_fpr(fpr_arr, tpr_arr, target):
        valid_indices = np.where(fpr_arr <= target)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            return tpr_arr[idx], fpr_arr[idx]
        else:
            return 0.0, 0.0

    tpr_5, fpr_5 = get_tpr_at_fpr(fpr, tpr, 0.05)
    tpr_1, fpr_1 = get_tpr_at_fpr(fpr, tpr, 0.01)
    tpr_01, fpr_01 = get_tpr_at_fpr(fpr, tpr, 0.001)
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"AUC: {auc:.4f}")
    print(f"TPR @ 5% FPR: {tpr_5:.4f} (Actual FPR: {fpr_5:.4f})")
    print(f"TPR @ 1% FPR: {tpr_1:.4f} (Actual FPR: {fpr_1:.4f})")
    print(f"TPR @ 0.1% FPR: {tpr_01:.4f} (Actual FPR: {fpr_01:.4f})")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate Lineage Detector")
    parser.add_argument('--config', type=str, default='configs/small_model/mobilenet_caltech.yaml', help='Path to config file')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only (skip training)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for evaluation')
    parser.add_argument('--data_dir', type=str, help='Path to test data directory (for evaluation)')
    parser.add_argument('--model_arch', type=str, default='mobilenet', choices=['mobilenet', 'resnet'], help='Model architecture (mobilenet or resnet)')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)
    device = get_device()

    # Evaluation Mode
    if args.evaluate:
        if args.checkpoint and args.data_dir:
            # Pre-trained Evaluation Mode
            evaluate_pretrained_model(args.checkpoint, args.data_dir, device, args.model_arch)
            return
        else:
            print("Error: --evaluate requires --checkpoint and --data_dir")
            return

    # Load Embeddings
    emb_dir = config['embeddings']['output_dir']
    
    try:
        child_data = torch.load(os.path.join(emb_dir, 'child_train_embeddings.pt'))
        ind_data = torch.load(os.path.join(emb_dir, 'independent_train_embeddings.pt'))
    except FileNotFoundError as e:
        print(f"Error loading embeddings: {e}")
        print("Please run generate_embeddings.py first.")
        return

    pos_features = child_data['embeddings']
    neg_features = ind_data['embeddings']
    
    # Create Labels (1 for Related/Child, 0 for Unrelated/Independent)
    pos_labels = torch.ones(len(pos_features), dtype=torch.long)
    neg_labels = torch.zeros(len(neg_features), dtype=torch.long)
    
    X = torch.cat([pos_features, neg_features], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)
    
    # Shuffle
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    
    print(f"Total samples: {len(X)} (Pos: {len(pos_features)}, Neg: {len(neg_features)})")

    # Split Train/Val
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}")
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model Setup
    input_dim = X.shape[1]
    hidden_dim = config['lineage_training'].get('hidden_dim', 512)
    model = LineageDetector(input_dim, hidden_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lineage_training'].get('learning_rate', 0.001))
    
    num_epochs = config['lineage_training'].get('num_epochs', 20)
    
    print(f"Training Lineage Detector for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        
        acc = accuracy_score(val_targets, val_preds)
        try:
            auc = roc_auc_score(val_targets, val_probs)
            
            # Calculate TPR at fixed FPR
            fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
            
            # Calculate resolution
            num_negatives = np.sum(np.array(val_targets) == 0)
            min_fpr_step = 1.0 / num_negatives if num_negatives > 0 else 0
            
            def get_tpr_at_fpr(fpr_arr, tpr_arr, target):
                # Find the last index where fpr <= target
                # Since fpr is increasing, this gives the highest TPR for that FPR constraint
                valid_indices = np.where(fpr_arr <= target)[0]
                if len(valid_indices) > 0:
                    idx = valid_indices[-1]
                    return tpr_arr[idx], fpr_arr[idx]
                else:
                    return 0.0, 0.0

            # Find TPR at FPR = 5%
            tpr_at_5_percent_fpr, actual_fpr_5 = get_tpr_at_fpr(fpr, tpr, 0.05)

            # Find TPR at FPR = 1%
            tpr_at_1_percent_fpr, actual_fpr_1 = get_tpr_at_fpr(fpr, tpr, 0.01)
            
            # Find TPR at FPR = 0.1%
            tpr_at_01_percent_fpr, actual_fpr_01 = get_tpr_at_fpr(fpr, tpr, 0.001)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {acc:.4f} - AUC: {auc:.4f}")
            print(f"    TPR @ 5% FPR: {tpr_at_5_percent_fpr:.4f} (Actual FPR: {actual_fpr_5:.4f})")
            print(f"    TPR @ 1% FPR: {tpr_at_1_percent_fpr:.4f} (Actual FPR: {actual_fpr_1:.4f})")
            if min_fpr_step > 0.01:
                 print(f"    (Note: Min FPR step is {min_fpr_step:.4f}, so 1% FPR requires 0 False Positives)")
            
        except ValueError:
            # Handle case where only one class is present in validation set
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {acc:.4f}")
            print("    Warning: AUC/TPR calculation failed (likely only one class in validation set)")

    # Final Detailed Report
    print("\n" + "="*50)
    print("Final Evaluation Report")
    print("="*50)
    
    cm = confusion_matrix(val_targets, val_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:")
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")
    print(f"Accuracy: {acc:.4f}")
    
    if 'auc' in locals():
        print(f"AUC: {auc:.4f}")
        print(f"TPR @ 5% FPR: {tpr_at_5_percent_fpr:.4f}")
        print(f"TPR @ 1% FPR: {tpr_at_1_percent_fpr:.4f}")
        print(f"TPR @ 0.1% FPR: {tpr_at_01_percent_fpr:.4f}")

    # Save Metrics
    metrics = {
        'accuracy': acc,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }
    if 'auc' in locals():
        metrics.update({
            'auc': auc,
            'tpr_at_5_percent_fpr': tpr_at_5_percent_fpr,
            'tpr_at_1_percent_fpr': tpr_at_1_percent_fpr,
            'tpr_at_01_percent_fpr': tpr_at_01_percent_fpr
        })

    # Save Model
    save_path = config['lineage_training']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metrics_path = os.path.join(os.path.dirname(save_path), 'lineage_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    torch.save(model.state_dict(), save_path)
    print(f"Lineage detector saved to {save_path}")

if __name__ == "__main__":
    main()
