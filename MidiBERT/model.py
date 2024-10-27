import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel, AlbertModel, RobertaModel, DistilBertModel

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e, model_name='bert'):
        super().__init__()
        
        if model_name == 'bert':
            self.bert = BertModel(bertConfig)
            self.hidden_size = bertConfig.hidden_size
            bertConfig.d_model = bertConfig.hidden_size
            self.bertConfig = bertConfig
        elif model_name == 'albert':
            self.bert = AlbertModel(bertConfig)
            self.hidden_size = bertConfig.hidden_size
            bertConfig.d_model = bertConfig.embedding_size
            self.bertConfig = bertConfig
        elif model_name == 'distilbert':
            self.bert = DistilBertModel(bertConfig)
            self.hidden_size = bertConfig.hidden_size
            bertConfig.d_model = bertConfig.hidden_size
            self.bertConfig = bertConfig

        # token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []      # [3,18,88,66]
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256] # [128, 128, 128, 128]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.long)
        
        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types 
        # 異なるトークンタイプの情報を融合するための線形層（次元をBERTのhidden_sizeに変換）
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)


    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1) # 異なるトークンタイプの情報を融合
        emb_linear = self.in_linear(embs) # 異なるトークンタイプの情報を融合したものを、BERTのhidden_sizeに変換

        # feed to bert 
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        #y = y.last_hidden_state         # (batch_size, seq_len, hidden_size)
        return y
    
    def get_rand_tok(self):
        c1,c2,c3,c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array([random.choice(range(c1)),random.choice(range(c2)),random.choice(range(c3)),random.choice(range(c4))])