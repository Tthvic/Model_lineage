import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TransformerEncoder(nn.Module):
    def __init__(self, feat_dim=1280, d_model=128, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.feat_proj = nn.Linear(1280,128)  # feature projection
        self.layer_norm = nn.LayerNorm(d_model)  # normalization for stability

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # positional encoding
        self.position_encoding = nn.Embedding(1000, d_model)  # set max sequence length to 1000

        # attentive global representation
        self.attn_pooling = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)

    def compute_valid_lengths(self, x):
        """Compute the valid length for each sample."""
        mask = (x.sum(dim=-1) != 0)  # [batch_size, seq_len]
        valid_lengths = mask.int().sum(dim=1)  # [batch_size]
        return valid_lengths

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        valid_lengths = self.compute_valid_lengths(x)
        # print(x.shape, "x_shape")
        # feature projection + normalization
        feat_proj = self.layer_norm(self.feat_proj(x))  # [batch_size, seq_len, d_model]

        # positional encoding
        pos_ids = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)  # [batch_size, seq_len]
        pos_emb = self.position_encoding(pos_ids)  # [batch_size, seq_len, d_model]

        # Transformer encoding
        inputs = feat_proj + pos_emb  # add positional information
        padding_mask = (x.sum(dim=-1) == 0)  # [batch_size, seq_len]
        outputs = self.transformer(inputs, src_key_padding_mask=padding_mask)  # [batch_size, seq_len, d_model]

        # attentive global representation (attentive pooling)
        query = torch.mean(outputs, dim=1, keepdim=True)  # mean as query
        global_vec, _ = self.attn_pooling(query, outputs, outputs,
                                          key_padding_mask=padding_mask)  # [batch_size, 1, d_model]
        global_vec = global_vec.squeeze(1)  # [batch_size, d_model]
        
        return global_vec

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