import copy
from typing import Dict, List

import torch
from torch import nn

from .moe import ConvBNAct, FusionMambaBlock, MoEBlock


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            ConvBNAct(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, act=False),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3, padding=1),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _collect_outputs(logits: torch.Tensor, feature: torch.Tensor, aux_list: List[Dict[str, torch.Tensor]]) -> Dict:
    balance = feature.new_tensor(0.0)
    routers = []
    for aux in aux_list:
        balance = balance + aux["balance_loss"]
        routers.append(aux["router_probs"])
    return {"logits": logits, "feature": feature, "router_probs": routers, "balance_loss": balance}


class TeacherHSMS(nn.Module):
    def __init__(
        self,
        hs_channels: int = 144,
        ms_channels: int = 10,
        num_classes: int = 15,
        base_channels: int = 64,
        hs_reduced_channels: int = 32,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hs_reduce = ConvBNAct(hs_channels, hs_reduced_channels, kernel_size=1, padding=0)
        self.hs_stem = Stem(hs_reduced_channels, base_channels)
        self.ms_stem = Stem(ms_channels, base_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNAct(base_channels * 2, base_channels, kernel_size=1, padding=0)
        self.fusion_mamba = FusionMambaBlock(base_channels)
        self.moe1 = MoEBlock(base_channels, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.context = ResidualConvBlock(base_channels, dilation=2)
        self.moe2 = MoEBlock(base_channels, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, hs: torch.Tensor, ms: torch.Tensor) -> Dict:
        hs_feat = self.hs_stem(self.hs_reduce(hs))
        ms_feat = self.ms_stem(ms)
        gate = self.gate(torch.cat([hs_feat, ms_feat], dim=1))
        gated = gate * hs_feat + (1.0 - gate) * ms_feat
        fused = self.fuse(torch.cat([gated, hs_feat + ms_feat], dim=1))
        fused = self.fusion_mamba(fused)
        x, aux1 = self.moe1(fused)
        x = self.context(x)
        x, aux2 = self.moe2(x)
        logits = self.classifier(x)
        return _collect_outputs(logits, x, [aux1, aux2])


class TeacherHSMSPlus(TeacherHSMS):
    """Compatibility wrapper that keeps old configs on the base teacher path."""

    def __init__(
        self,
        hs_channels: int = 144,
        ms_channels: int = 10,
        num_classes: int = 15,
        base_channels: int = 64,
        hs_reduced_channels: int = 32,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        norm: str = "group",
        spectral_groups: int = 8,
    ) -> None:
        _ = (norm, spectral_groups)
        super().__init__(
            hs_channels=hs_channels,
            ms_channels=ms_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            hs_reduced_channels=hs_reduced_channels,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )


class StudentMSMoE(nn.Module):
    def __init__(
        self,
        ms_channels: int = 10,
        num_classes: int = 15,
        base_channels: int = 64,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ms_stem = Stem(ms_channels, base_channels)
        self.hallucinate_hs = nn.Sequential(
            ResidualConvBlock(base_channels),
            ConvBNAct(base_channels, base_channels, kernel_size=1, padding=0),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNAct(base_channels * 2, base_channels, kernel_size=1, padding=0)
        self.fusion_mamba = FusionMambaBlock(base_channels)
        self.moe1 = MoEBlock(base_channels, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.context = ResidualConvBlock(base_channels, dilation=2)
        self.moe2 = MoEBlock(base_channels, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, ms: torch.Tensor) -> Dict:
        ms_feat = self.ms_stem(ms)
        hs_like = self.hallucinate_hs(ms_feat)
        gate = self.gate(torch.cat([ms_feat, hs_like], dim=1))
        fused = gate * hs_like + (1.0 - gate) * ms_feat
        x = self.fuse(torch.cat([fused, ms_feat + hs_like], dim=1))
        x = self.fusion_mamba(x)
        x, aux1 = self.moe1(x)
        x = self.context(x)
        x, aux2 = self.moe2(x)
        logits = self.classifier(x)
        return _collect_outputs(logits, x, [aux1, aux2])


class MSBaseline(nn.Module):
    def __init__(self, ms_channels: int = 10, num_classes: int = 15, base_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            Stem(ms_channels, base_channels),
            ResidualConvBlock(base_channels, dilation=1),
            ResidualConvBlock(base_channels, dilation=2),
            ResidualConvBlock(base_channels, dilation=3),
        )
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, ms: torch.Tensor) -> Dict:
        feature = self.encoder(ms)
        logits = self.classifier(feature)
        return {
            "logits": logits,
            "feature": feature,
            "router_probs": [],
            "balance_loss": feature.new_tensor(0.0),
        }


def build_teacher_model(config: Dict, hs_channels: int, ms_channels: int) -> nn.Module:
    model_cfg = config["model"]
    common = dict(
        hs_channels=hs_channels,
        ms_channels=ms_channels,
        num_classes=config["data"]["num_classes"],
        base_channels=model_cfg["base_channels"],
        hs_reduced_channels=model_cfg["hs_reduced_channels"],
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg["top_k"],
        dropout=model_cfg["dropout"],
    )
    return TeacherHSMS(**common)


def teacher_config_from_checkpoint(current_config: Dict, checkpoint: Dict) -> Dict:
    build_config = copy.deepcopy(checkpoint.get("config", current_config))
    build_config.setdefault("data", {}).update(
        {
            "num_classes": current_config["data"]["num_classes"],
        }
    )
    build_config.setdefault("model", {})
    model_name = checkpoint.get("model_name", "")
    if model_name == "TeacherHSMS":
        build_config["model"]["teacher_variant"] = "base"
    elif model_name == "TeacherHSMSPlus":
        build_config["model"]["teacher_variant"] = "base"

    for key, value in current_config.get("model", {}).items():
        build_config["model"].setdefault(key, value)
    return build_config
