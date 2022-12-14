import math
import torch
import torch.nn as nn

from einops import rearrange

# #################################### Positional Encoding ######################################

class PositionalEncoding(nn.Module):

    def __init__(self, num_harmonic: int = 60, concat_axis=-1):
        super(PositionalEncoding, self).__init__()

        self.num_harmonic = num_harmonic

    def forward(self, x):

        emb = [x]

        for i in range(self.num_harmonic):
            emb.append(torch.sin((2.0 ** i) * x))
            emb.append(torch.cos((2.0 ** i) * x))

        emb = torch.cat(emb, -1)
        return emb