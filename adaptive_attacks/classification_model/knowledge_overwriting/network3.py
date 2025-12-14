import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

DIM=32
class TransformerEncoder(nn.Module):
    def __init__(self, feat_dim=512, d_model=DIM, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.fc1= nn.Linear(1280, d_model) 
        # self.feat_proj = nn.Linear(feat_dim, d_model) 
        # self.layer_norm = nn.LayerNorm(d_model)
        # # Transformer
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True, norm_first=True
        # )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # #
        # self.position_encoding = nn.Embedding(1000, d_model) 

        # # 
        # self.attn_pooling = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)

    def compute_valid_lengths(self, x):
        mask = (x.sum(dim=-1) != 0)  # [batch_size, seq_len]
        valid_lengths = mask.int().sum(dim=1)  # [batch_size]
        return valid_lengths

    def forward(self, x):
        # batch_size, seq_len, feat_dim = x.shape
        # valid_lengths = self.compute_valid_lengths(x)

        # # 
        # feat_proj = self.layer_norm(self.feat_proj(x))  # [batch_size, seq_len, d_model]

        # #
        # pos_ids = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)  # [batch_size, seq_len]
        # pos_emb = self.position_encoding(pos_ids)  # [batch_size, seq_len, d_model]

        # #
        # inputs = feat_proj + pos_emb  
        # padding_mask = (x.sum(dim=-1) == 0)  # [batch_size, seq_len]
        # outputs = self.transformer(inputs, src_key_padding_mask=padding_mask)  # [batch_size, seq_len, d_model]

        # # 
        # query = torch.mean(outputs, dim=1, keepdim=True)  
        # global_vec, _ = self.attn_pooling(query, outputs, outputs,
        #                                   key_padding_mask=padding_mask)  # [batch_size, 1, d_model]
        # global_vec = global_vec.squeeze(1)  # [batch_size, d_model]
        #
        global_vec=torch.mean(self.fc1(x),dim=1)
        # global_vec=torch.mean(x,dim=1)
        return global_vec






class VectorRelationNet(nn.Module):
    def __init__(self, embedding_dim):
        super(VectorRelationNet, self).__init__()
        self.fc1 = nn.Linear(DIM * 2, DIM) 
        self.relu=nn.ReLU()
        # self.dropout = nn.Dropout(p=0.3)
        # self.fc3 = nn.Linear(128, 128)
    def forward(self, b_afea,cfea):

        h1 = self.fc1(torch.cat([b_afea, cfea], dim=-1))
        h1=self.relu(h1)
 
        return h1

