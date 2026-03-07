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

        # Expand dvec to (N, M, N, D)
        dvecs = input.unsqueeze(2).expand(n_speakers, n_utter, n_speakers, d_emb) # Have same N vector in dim 2 

        # Build normal centroids from eq. (1)
        # Make centroid shaped like (N * M, N, D)
        centroids = input.mean(dim=1).to(input.device) # (N, D)
        centroids = centroids.unsqueeze(0).expand(n_speakers * n_utter, n_speakers, d_emb) # (N * M, N, D)
        centroids = centroids.reshape(-1, d_emb) # (N * M * N, D)

        # Modified centroids based on eq. (8)
        mod_cent = torch.cat([input[:, 1:, :], input[:, :-1, :]], dim=1) # Make a long array from index 1 to -2 in dimension M
        mod_cent = mod_cent.unfold(dimension=1, size=(n_utter - 1), step=1) # (N, M, D, M - 1)
        mod_cent = mod_cent.mean(dim=-1) # (N, M, D)
        mod_cent = mod_cent.reshape(-1, d_emb) # (N * M, D)

        # Replace centroids w/ mod_cent, when j (dim 0) == k (dim 2)
        # need to replace n_utter vectors for every speaker (hence n_utter * n_speakers)
        # then need to add offset for every speaker, so reshape to a shape of (n_speakers, n_utter)
        # as our centroid shape is (N * M * N, D) that means every for one speaker and an utterance
        # it has n_speaker amount of rows. Need to replace only current speaker for that n_speaker rows.
        indices = torch.arange(n_utter * n_speakers).reshape(n_speakers, -1) * n_speakers \
            + torch.arange(n_speakers).unsqueeze(-1) # offset by the index of speaker
        indices = indices.reshape(-1).to(input.device)
        transformed_cent = centroids.index_copy(0, indices, mod_cent)
        transformed_cent = transformed_cent.view_as(dvecs)

        # Plug into CosineSimilarity
        # Should be calculate similarity between N vectors of length D (N, M, N, D)
        return F.cosine_similarity(dvecs, transformed_cent, 3, 1e-6)

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
    


        
