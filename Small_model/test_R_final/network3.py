import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch
# import torch.nn as nn

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class GRUEncoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=128, num_layers=2):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim) 
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)

    def compute_valid_lengths(self, x):
        mask = (x.sum(dim=-1) != 0)  # [batch_size, seq_len]，
        valid_lengths = mask.int().sum(dim=1)  # [batch_size]
        return valid_lengths

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        valid_lengths = self.compute_valid_lengths(x)  # [batch_size]

        feat_proj = self.feat_proj(x)  # [batch_size, seq_len, hidden_dim]

        sorted_lengths, sorted_idx = valid_lengths.sort(descending=True)
        sorted_feat_proj = feat_proj[sorted_idx]

        packed_inputs = pack_padded_sequence(sorted_feat_proj, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, _ = self.gru(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)  # [batch_size, seq_len, hidden_dim]

        global_vecs = outputs[torch.arange(batch_size), sorted_lengths - 1]  # [batch, hidden_dim]

        _, original_idx = sorted_idx.sort()
        global_vecs = global_vecs[original_idx]

        return global_vecs


# import torch
# import torch.nn as nn


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
class TransformerEncoder(nn.Module):
    # class SimpleMLPEncoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=128, output_dim=128, dropout=0.1):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)  # feature projection
        self.layer_norm = nn.LayerNorm(hidden_dim)  # normalization
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # MLP layer
        # self.fc2 = nn.Linear(hidden_dim, output_dim)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, seq_len, feat_dim]
        """
        x = self.feat_proj(x)  # linear projection [batch_size, seq_len, hidden_dim]
        x = self.layer_norm(x)  # normalization for stability

        # global average pooling (RNN/Transformer alternative)
        x = x.mean(dim=1)  # [batch_size, hidden_dim]

        x = F.relu(self.fc1(x))  # activation
        x = self.dropout(x)  # dropout to prevent overfitting
        # x = self.fc2(x) 

        return x  # [batch_size, output_dim]



class VectorRelationNet(nn.Module):
    def __init__(self, embedding_dim):
        super(VectorRelationNet, self).__init__()
        self.fc1 = nn.Linear(128 * 2, 128)  # input is concatenation of two vectors
        self.relu=nn.ReLU()
        # self.fc3 = nn.Linear(128, 128)
    def forward(self, b_afea,cfea):
        # concatenate two predicted vectors
        h1 = self.fc1(torch.cat([b_afea, cfea], dim=-1))
        h1=self.relu(h1)
        return h1
