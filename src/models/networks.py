import copy
from typing import Dict, List

import torch
from torch import nn

from .moe import ConvBNAct, MoEBlock


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


class SpectralMambaGate(nn.Module):
    def __init__(
        self,
        hs_channels: int,
        mamba_dim: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "hs_spectral_type='mamba' requires the real mamba_ssm package. "
                "Install mamba_ssm in the training environment before training the teacher."
            ) from exc

        self.hs_channels = hs_channels
        self.in_proj = nn.Linear(1, mamba_dim)
        self.mamba = Mamba(d_model=mamba_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate_proj = nn.Linear(mamba_dim, 1)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        if hs.shape[1] != self.hs_channels:
            raise ValueError(f"Expected HS input with {self.hs_channels} channels, got {hs.shape[1]}.")
        spectral_sequence = hs.mean(dim=(2, 3)).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_proj(self.mamba(self.in_proj(spectral_sequence))))
        return hs * gate.unsqueeze(-1)


def _build_hs_spectral_gate(
    hs_spectral_type: str,
    hs_channels: int,
    hs_mamba_dim: int,
    hs_mamba_d_state: int,
    hs_mamba_d_conv: int,
    hs_mamba_expand: int,
) -> nn.Module:
    spectral_type = str(hs_spectral_type or "none").lower()
    if spectral_type in {"none", "identity", "off"}:
        return nn.Identity()
    if spectral_type == "mamba":
        return SpectralMambaGate(
            hs_channels=hs_channels,
            mamba_dim=hs_mamba_dim,
            d_state=hs_mamba_d_state,
            d_conv=hs_mamba_d_conv,
            expand=hs_mamba_expand,
        )
    raise ValueError(f"Unsupported hs_spectral_type: {hs_spectral_type}")


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
        hs_spectral_type: str = "none",
        hs_mamba_dim: int = 16,
        hs_mamba_d_state: int = 16,
        hs_mamba_d_conv: int = 4,
        hs_mamba_expand: int = 2,
    ) -> None:
        super().__init__()
        self.hs_spectral = _build_hs_spectral_gate(
            hs_spectral_type=hs_spectral_type,
            hs_channels=hs_channels,
            hs_mamba_dim=hs_mamba_dim,
            hs_mamba_d_state=hs_mamba_d_state,
            hs_mamba_d_conv=hs_mamba_d_conv,
            hs_mamba_expand=hs_mamba_expand,
        )
        self.hs_reduce = ConvBNAct(hs_channels, hs_reduced_channels, kernel_size=1, padding=0)
        self.hs_stem = Stem(hs_reduced_channels, base_channels)
        self.ms_stem = Stem(ms_channels, base_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNAct(base_channels * 2, base_channels, kernel_size=1, padding=0)
        self.moe1 = MoEBlock(base_channels, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.context = ResidualConvBlock(base_channels, dilation=2)
        self.moe2 = MoEBlock(base_channels, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, hs: torch.Tensor, ms: torch.Tensor) -> Dict:
        hs = self.hs_spectral(hs)
        hs_feat = self.hs_stem(self.hs_reduce(hs))
        ms_feat = self.ms_stem(ms)
        gate = self.gate(torch.cat([hs_feat, ms_feat], dim=1))
        gated = gate * hs_feat + (1.0 - gate) * ms_feat
        fused = self.fuse(torch.cat([gated, hs_feat + ms_feat], dim=1))
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
        hs_spectral_type: str = "none",
        hs_mamba_dim: int = 16,
        hs_mamba_d_state: int = 16,
        hs_mamba_d_conv: int = 4,
        hs_mamba_expand: int = 2,
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
            hs_spectral_type=hs_spectral_type,
            hs_mamba_dim=hs_mamba_dim,
            hs_mamba_d_state=hs_mamba_d_state,
            hs_mamba_d_conv=hs_mamba_d_conv,
            hs_mamba_expand=hs_mamba_expand,
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
        hs_spectral_type=model_cfg.get("hs_spectral_type", "none"),
        hs_mamba_dim=model_cfg.get("hs_mamba_dim", 16),
        hs_mamba_d_state=model_cfg.get("hs_mamba_d_state", 16),
        hs_mamba_d_conv=model_cfg.get("hs_mamba_d_conv", 4),
        hs_mamba_expand=model_cfg.get("hs_mamba_expand", 2),
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
