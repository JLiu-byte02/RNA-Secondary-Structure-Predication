"""Reuse unchanged E models and the original head-only B0 baseline."""
import torch
from torch import nn
from src.pair_head import PairHead
from scripts.run_thermo_fast import ThermoFusion

class B0(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=PairHead()

    def forward(self,embedding,physical,distance,valid_mask,return_initial=False):
        initial=self.head(embedding).float()
        if not torch.isfinite(initial).all():raise FloatingPointError('Nonfinite B0 logits')
        logits=initial.masked_fill(~valid_mask,-100.)
        return (logits,initial,torch.zeros_like(initial)) if return_initial else logits

def make_model(mode):
    return B0() if mode=='B0' else ThermoFusion(mode)
