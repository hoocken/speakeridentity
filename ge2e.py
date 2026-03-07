# Adapted from https://github.com/yistLin/dvector/blob/master/modules/ge2e.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GE2E(nn.Module):
    """
    Based on https://arxiv.org/pdf/1710.10467.
    Accepts a tensor of shape (N, M, D) where
        N is the number of speakers,
        M is the number of utterances
        D is the dvector dimension. 
    """
    def __init__(self, init_w=10.0, init_b =-5.0, loss_method="softmax"):
        super(GE2E, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([init_w]))
        self.b = nn.Parameter(torch.FloatTensor([init_b]))

        if loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax 
        elif loss_method == "contrast":
            self.embed_loss = self.embed_loss_constrast 

    def cosine_similarity(self, input: Tensor):
        """
        Calculates the cosine similarity matrix of the d-vector e_ji with
        the centroid of utterances by another speaker k. Here, if the speakers j, k
        are the same, then for the similarity cos_jik we use the modified centroid 
        which is the mean of all embeddings except for the embedding i.
        https://en.wikipedia.org/wiki/Cosine_similarity
        
        :param input: D-vectors
        :type input: Tensor of shape (N, M, D)
        
        :returns S: Similarity matrix of shape (N, M, N) 
        """
        n_speakers, n_utter, d_emb = input.shape

        # dvecs: (N, M, N, D) - each e_ji compared to centroids of k
        dvecs = input.unsqueeze(2).expand(n_speakers, n_utter, n_speakers, d_emb)

        # regular centroids (N, D)
        centroids = input.mean(dim=1)  # (N, D)

        # modified centroids excluding each utterance: (N, M, D)
        sum_over = input.sum(dim=1, keepdim=True)  # (N, 1, D)
        mod_cent = (sum_over - input) / (n_utter - 1)  # (N, M, D)

        # expand to (N, M, N, D)
        cent_k = centroids.unsqueeze(0).unsqueeze(0).expand(n_speakers, n_utter, n_speakers, d_emb)
        mod_k = mod_cent.unsqueeze(2).expand(n_speakers, n_utter, n_speakers, d_emb)

        # mask where speaker j == k (shape (N, M, N, 1))
        j_idx = torch.arange(n_speakers, device=input.device).view(n_speakers, 1, 1)
        k_idx = torch.arange(n_speakers, device=input.device).view(1, 1, n_speakers)
        mask = (j_idx == k_idx).unsqueeze(-1).expand(-1, n_utter, -1, d_emb)

        # choose modified centroid when j==k, else regular centroid
        transformed_cent = torch.where(mask, mod_k, cent_k)

        # cosine similarity over last dim
        return F.cosine_similarity(dvecs, transformed_cent, dim=-1, eps=1e-6)

    def embed_loss_softmax(self, input: Tensor):
        """
        Calculates the softmax loss according to the formula below:

        $$
        L(e_{ji}) = - S_{ji,j} + log sum_{k = 1}^{N} exp(S_{ji,k})
        $$
        
        :param input: Calculated similarity **S**
        :type input: Tensor of shape (N, M, N)
        """
        n_speakers, _, _ = input.shape
        diags = input[torch.arange(n_speakers).to(input.device), :, torch.arange(n_speakers).to(input.device)] # (N, M)
        log_exp = torch.logsumexp(input, dim=-1) # (N, M)
        return -diags + log_exp

    def embed_loss_constrast(self, input: Tensor):
        """
        Calculates the contrast loss according to the following formula:

        $$
        L(e_{ji}) = 1 - sigma(S_{ji,j}) + underset{max}{1 leq k leq N, k neq j} sigma(S_{ji, k})
        $$
        
        :param input: Calculated similarity **S**
        :type input: Tensor of shape (N, M, N)
        """
        n_speakers, n_utter, _ = input.shape
        input = torch.sigmoid(input)
        diags = input[torch.arange(n_speakers).to(input.device), :, torch.arange(n_speakers).to(input.device)] # (N, M)

        cat = torch.cat([input[:, :, 1:], input[:, :, :-1]], 2)
        unfold = cat.unfold(2, n_speakers - 1, 1) # (N, M, N, N - 1)
        unfold = unfold.reshape(-1, n_speakers - 1) # (N * M * N, N - 1)
        
        indices = torch.arange(n_utter * n_speakers).reshape(n_speakers, -1) * n_speakers \
            + torch.arange(n_speakers).unsqueeze(-1) # offset by the index of speaker
        indices = indices.reshape(-1).to(input.device)

        trunc = unfold[indices, :] # (N * M, N - 1)
        trunc = trunc.reshape(n_speakers, n_utter, n_speakers - 1) # (N, M, N-1)
        max, _ = trunc.max(dim=-1) # (N, M)

        return 1 - diags + max

    def forward(self, input: Tensor):
        """
        Calculates the similarity matrix and then applies an
        affine transformation on it.
        
        :param input: D-vectors
        :type input: Tensor of shape (N, M, D)
        :return loss: Total loss calculated through chosen `loss_method`
        """
        # torch.clamp(self.w, 1e-6)
        calc = self.w * self.cosine_similarity(input) + self.b
        return self.embed_loss(calc).sum()
    


        
