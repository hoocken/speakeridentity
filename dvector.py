# https://arxiv.org/pdf/1710.10467

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

class D_VECTOR(nn.Module):
    def __init__(self, num_layers=3, dim_input=80, dim_cell=256, dim_emb=6, seg_len=160):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell,
                            num_layers=num_layers, batch_first=True) # Sets dim to (Batch, Seq, Feature)
        self.embedding = nn.Linear(in_features=dim_cell, out_features=dim_emb)
        self.seg_len = seg_len
        
    def forward(self, x):
        """
        Calculates the dvector for a batch of utterances.
        It first passes the batch through an LSTM, then the last
        result sequence is passed through a linear layer. Lastly, it is 
        normalized.
        
        :param x: Tensor of shape (Batch, Sequence Length, Input Dimension)
        :returns dvector: D-Vector of shape (Batch, Output Dimension)
        """   
        print(x.shape)
        lstm_outs, _ = self.lstm(x) # (Batch, Seq, 1 * hidden_size)
        embedding_vector = self.embedding(lstm_outs[:, -1, :]) # Take output of last sequence, now (Batch, Dim Emb)
        norm = embedding_vector.norm(p=2, dim=-1, keepdim=True)
        return embedding_vector / norm # Normalized dvector
    
    @torch.compile(fullgraph=True)
    def embed_utterance(self, x):
        """
        Embed utterance using sliding window method of length 160

        :param x: Tensor of shape (1, seg_length, mel_dim)
        :returns dvector: D-Vector of shape (output dimension)
        """
        step = self.seg_len // 2
        x = torch.squeeze(x).transpose(0, 1)
        pad = torch.cat([x, torch.zeros([step - x.shape[0] % step, x.shape[1]])], 0)
        segments = pad.unfold(0, self.seg_len, step)
        segments = segments.transpose(1, 2)

        embeddings = self.forward(segments)
        embed = embeddings.mean(dim=0)
        embed = embed.div(embed.norm(p=2, dim=-1, keepdim=True)) # Normalize
        return embed
    
    @torch.compile(fullgraph=True)
    def embed_utterances(self, x):
        """
        Embed utterances

        :param x: Tensor of shape (batch, seg_length, mel_dim)
        """
        embed = torch.stack([self.embed_utterance(uttr) for uttr in x]).mean(dim=0)
        return embed / (embed.norm(p=2, dim=-1, keepdim=True))
    