from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class LocalConvExpert(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedContextExpert(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, padding=2, dilation=2),
            ConvBNAct(channels, channels, kernel_size=3, padding=3, dilation=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelMLPExpert(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DirectionalSequenceExpert(nn.Module):
    """A dependency-light directional sequence expert inspired by spatial SSM blocks."""

    def __init__(self, channels: int, kernel_size: int = 9) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.h_conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        horizontal = x.permute(0, 2, 1, 3).reshape(bsz * height, channels, width)
        horizontal = self.h_conv(horizontal).reshape(bsz, height, channels, width).permute(0, 2, 1, 3)

        vertical = x.permute(0, 3, 1, 2).reshape(bsz * width, channels, height)
        vertical = self.v_conv(vertical).reshape(bsz, width, channels, height).permute(0, 2, 3, 1)
        return self.proj(x + horizontal + vertical)


class SparseMoE(nn.Module):
    def __init__(self, channels: int, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        if num_experts != 4:
            raise ValueError("This implementation uses exactly four named experts.")
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Conv2d(channels, num_experts, kernel_size=1)
        self.experts = nn.ModuleList(
            [
                LocalConvExpert(channels),
                DilatedContextExpert(channels),
                ChannelMLPExpert(channels),
                DirectionalSequenceExpert(channels),
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        router_logits = self.router(x)
        probs = F.softmax(router_logits, dim=1)
        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=1)
        sparse_weights = torch.zeros_like(probs).scatter_(1, top_indices, top_values)
        sparse_weights = sparse_weights / sparse_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (expert_outputs * sparse_weights.unsqueeze(2)).sum(dim=1)

        importance = probs.mean(dim=(0, 2, 3))
        load = (sparse_weights > 0).float().mean(dim=(0, 2, 3))
        balance_loss = self.num_experts * torch.sum(importance * load)
        aux = {
            "router_logits": router_logits,
            "router_probs": probs,
            "router_mask": sparse_weights,
            "balance_loss": balance_loss,
        }
        return mixed, aux


class MoEBlock(nn.Module):
    def __init__(self, channels: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.moe = SparseMoE(channels, num_experts=num_experts, top_k=top_k)
        self.proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.drop = nn.Dropout2d(dropout)
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        routed, aux = self.moe(self.norm(x))
        x = x + self.drop(self.proj(routed))
        x = x + self.drop(self.ffn(x))
        return x, aux
