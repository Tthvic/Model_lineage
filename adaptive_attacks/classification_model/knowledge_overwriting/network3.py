import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

DIM=32
class TransformerEncoder(nn.Module):
    def __init__(self, feat_dim=512, d_model=DIM, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.fc1= nn.Linear(1280, d_model) 
        # self.feat_proj = nn.Linear(feat_dim, d_model)  # 特征投影
        # self.layer_norm = nn.LayerNorm(d_model)  # 归一化，提升稳定性
        # # Transformer 编码层
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True, norm_first=True
        # )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # # 位置编码
        # self.position_encoding = nn.Embedding(1000, d_model)  # 设定最大序列长度为 1000

        # # 自适应全局表示
        # self.attn_pooling = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)

    def compute_valid_lengths(self, x):
        """ 计算每个样本的有效长度 """
        mask = (x.sum(dim=-1) != 0)  # [batch_size, seq_len]
        valid_lengths = mask.int().sum(dim=1)  # [batch_size]
        return valid_lengths

    def forward(self, x):
        # batch_size, seq_len, feat_dim = x.shape
        # valid_lengths = self.compute_valid_lengths(x)

        # # 特征投影 + 归一化
        # feat_proj = self.layer_norm(self.feat_proj(x))  # [batch_size, seq_len, d_model]

        # # 位置编码
        # pos_ids = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)  # [batch_size, seq_len]
        # pos_emb = self.position_encoding(pos_ids)  # [batch_size, seq_len, d_model]

        # # Transformer 编码
        # inputs = feat_proj + pos_emb  # 加入位置信息
        # padding_mask = (x.sum(dim=-1) == 0)  # [batch_size, seq_len]
        # outputs = self.transformer(inputs, src_key_padding_mask=padding_mask)  # [batch_size, seq_len, d_model]

        # # 自适应全局表示（Attentive Pooling）
        # query = torch.mean(outputs, dim=1, keepdim=True)  # 计算均值作为 query
        # global_vec, _ = self.attn_pooling(query, outputs, outputs,
        #                                   key_padding_mask=padding_mask)  # [batch_size, 1, d_model]
        # global_vec = global_vec.squeeze(1)  # [batch_size, d_model]
        #消融实验
        global_vec=torch.mean(self.fc1(x),dim=1)
        # global_vec=torch.mean(x,dim=1)
        return global_vec



# class TransformerEncoder(nn.Module):
#     def __init__(self, feat_dim=512, d_model=128, nhead=2, num_layers=2):
#         super().__init__()
#         self.feat_proj = nn.Linear(feat_dim, d_model)  # 特征投影

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=128
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def compute_valid_lengths(self, x):
#         """
#         计算每个样本的有效长度（去除填充部分）。
#         x: [batch_size, seq_len, feat_dim]
#         return: valid_lengths [batch_size]，表示每个样本的真实长度（不含填充）。
#         """
#         # 计算非零 token 的 mask
#         mask = (x.sum(dim=-1) != 0)  # [batch_size, seq_len]，非零位置为 True

#         # 找到最后一个非零索引（即有效序列长度）
#         valid_lengths = mask.int().sum(dim=1)  # [batch_size]
#         return valid_lengths

#     def forward(self, x):
#         batch_size, seq_len, feat_dim = x.shape
#         valid_lengths = self.compute_valid_lengths(x)  # 自动计算有效长度

#         feat_proj = self.feat_proj(x)  # [batch_size, seq_len, d_model]

#         # 生成 padding mask
#         padding_mask = (x.sum(dim=-1) == 0)  # [batch_size, seq_len]

#         # 转换形状以适配 Transformer
#         inputs = feat_proj.transpose(0, 1)  # [seq_len, batch, d_model]

#         outputs = self.transformer(inputs, src_key_padding_mask=padding_mask)  # [seq_len, batch, d_model]

#         # 取最后一个有效 token 作为 global_vec
#         global_vecs = []
#         for i in range(batch_size):
#             valid_idx = valid_lengths[i] - 1  # 真实数据的最后一个 index
#             global_vecs.append(outputs[valid_idx, i])  # 取对应的 global_vec

#         global_vecs = torch.stack(global_vecs, dim=0)  # [batch, d_model]
#         return global_vecs


class VectorRelationNet(nn.Module):
    def __init__(self, embedding_dim):
        super(VectorRelationNet, self).__init__()
        self.fc1 = nn.Linear(DIM * 2, DIM)  # 输入是三个向量拼接
        # self.fc1 = nn.Linear(1280 * 2, 1280)  # 输入是三个向量拼接
        self.relu=nn.ReLU()
        # self.dropout = nn.Dropout(p=0.3)
        # self.fc3 = nn.Linear(128, 128)
    def forward(self, b_afea,cfea):
        # 拼接两个预测向量
        # h1 = self.fc1(torch.cat([b_afea, cfea], dim=-1))
        h1 = self.fc1(torch.cat([b_afea, cfea], dim=-1))
        h1=self.relu(h1)
        # h1 = b_afea+ cfea
        # h1=b_afea*cfea
        # h1=self.relu(h1)
        # self.h1=self.dropout(h1)
        return h1

    # def forward(self, b_afea,cfea,parent):
    #     # 拼接两个预测向量
    #     h1 = self.fc1(torch.cat([b_afea, cfea], dim=-1))
    #     v_predicted = self.fc3(parent)  # 输出预测向量
    #     return h1,v_predicted