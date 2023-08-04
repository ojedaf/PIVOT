import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from torch import Tensor
import copy
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        print('max_len: ',max_len)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pos_emb = self.pe[:x.size(1)]
        pos_emb = pos_emb.squeeze(1).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = x + pos_emb
        return x
        
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
        
class TransformerEncoderLayer(nn.Module):
    
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Temporal_Module(nn.Module):
    def __init__(self, device, conf):
        super().__init__()
        
        self.device = device
        dim_model = conf['dim_model']
        nhead = conf['nhead']
        num_layers = conf['num_layers']
        self.num_segments = conf['num_segments']
        
        self.positional_encoder = PositionalEncoding(d_model = dim_model, max_len = self.num_segments + 1)
        encoder_layer = TransformerEncoderLayer(d_model=dim_model, nhead=nhead, batch_first=True, 
                                                norm_first = True)
        self.embedding = nn.Embedding(1, dim_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, videos_features, promptModule = None, type_task = 'TIL'):
        idx = torch.tensor([0]*videos_features.size(0)).to(self.device)
        cls_emb = self.embedding(idx)
        cls_emb = torch.unsqueeze(cls_emb, 1)

        videos_features = torch.cat((cls_emb, videos_features), 1)
        videos_features = self.positional_encoder(videos_features)
        if promptModule != None:
            videos_features = promptModule(videos_features, 'temp', type_task)

        videos_features = self.transformer_encoder(videos_features)

        if promptModule != None:
            L_tp = promptModule.num_sel_prompts*promptModule.L_tp_gn if promptModule.type_prompt == 'general' else promptModule.num_sel_prompts*promptModule.L_tp_tk
            videos_features = videos_features[:, :L_tp, :]
            context_emb = torch.mean(videos_features, dim = 1)
        else:
            context_emb = videos_features[:, 0, :]
        return context_emb