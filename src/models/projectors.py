from typing import Optional

from torch import nn


def _valid_group_count(channels: int, requested_groups: int) -> int:
    if requested_groups <= 0:
        raise ValueError("norm_groups must be positive when using group norm.")
    for groups in range(min(channels, requested_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _make_norm(norm: str, channels: int, norm_groups: int) -> nn.Module:
    norm = norm.lower()
    if norm == "group":
        return nn.GroupNorm(_valid_group_count(channels, norm_groups), channels)
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "layer":
        return nn.GroupNorm(1, channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported projector norm '{norm}'.")


class DenseAnySatProjector2d(nn.Module):
    """Dense 2D variant of AnySat's modality-specific projector idea."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        hidden_ratio: float = 2.0,
        norm: str = "group",
        norm_groups: int = 8,
        dropout: float = 0.0,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        hidden_channels = int(round(out_channels * hidden_ratio))
        if hidden_channels <= 0:
            raise ValueError("hidden_ratio must produce at least one hidden channel.")

        use_bias = bias if bias is not None else norm.lower() in {"none", "identity"}
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=use_bias),
            _make_norm(norm, out_channels, norm_groups),
            nn.GELU(),
            nn.Conv2d(out_channels, hidden_channels, kernel_size=1, padding=0, bias=use_bias),
            _make_norm(norm, hidden_channels, norm_groups),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0, bias=use_bias),
            _make_norm(norm, out_channels, norm_groups),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)
