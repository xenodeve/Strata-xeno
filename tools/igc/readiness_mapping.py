"""Frozen diagnostic readiness architecture; no language-model updates."""
import torch


class ReadinessMapper(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(2560,128),torch.nn.GELU(),torch.nn.Linear(128,2))
        self.register_buffer('mean',torch.zeros(2560))
        self.register_buffer('scale',torch.ones(2560))

    def forward(self,mixed):
        if mixed.ndim!=2 or mixed.shape[1]!=2560:raise ValueError('readiness activation shape')
        return super().forward((mixed-self.mean)/self.scale)
