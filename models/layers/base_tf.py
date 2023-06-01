import torch
from torch import nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from utils.utils import weight_init

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dim,h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, input_dim)
        self.attention = Attention()
        weight_init(self.linear_layers)
        nn.init.xavier_normal_(self.output_linear.weight)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(q, k, v, mask=mask, dropout=self.dropout)   #在这里使用原始embeding 而不是v来映射
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class AutoIntBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.relu = nn.ReLU()
        self.attention = MultiHeadedAttention(
                            input_dim = self.args.model_params['embedding_dim'],
                            h=self.args.model_params['n_heads'], 
                            d_model=self.args.model_params['att_hidden_dim'],
                            dropout=self.args.model_params['att_dropout'])
        self.dropout = nn.Dropout(p=self.args.model_params['att_dropout'])

    def forward(self, x):
        out = self.attention.forward(x, x, x, mask=None)
        if self.args.model_params['use_res']:
            out += x
        return self.dropout(self.relu(out))
    
class Encoderblock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = MultiHeadedAttention(h=self.args.n_heads, d_model=self.args.hidden_dim,dropout=self.args.dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=self.args.hidden_dim, d_ff=self.args.hidden_dim*4, dropout=self.args.dropout)
        self.input_sublayer = SublayerConnection(size=self.args.hidden_dim, dropout=self.args.dropout)
        self.output_sublayer = SublayerConnection(size=self.args.hidden_dim, dropout=self.args.dropout)
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            # zero_vec = -9e15*torch.ones_like(scores.transpose(0,1))
            # tmp = torch.where(mask == 0, zero_vec, scores.transpose(0,1))
            # scores = tmp.transpose(0,1)
            mask_ = mask.unsqueeze(1)
            mask_ = mask_.unsqueeze(1)
            scores = scores.masked_fill(mask_ == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

