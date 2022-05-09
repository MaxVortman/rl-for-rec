import torch
import torch.nn as nn


class EmbeddingDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = super(EmbeddingDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1).squeeze(2)
        return x
