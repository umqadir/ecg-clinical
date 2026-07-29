"""Independent implementations of the two preregistered ECG architectures."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    activate: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm1d(out_channels),
    ]
    if activate:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class XResNetBottleneck1d(nn.Module):
    """One-dimensional bottleneck with an anti-aliased xResNet shortcut."""

    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.body = nn.Sequential(
            _conv_bn_relu(in_channels, channels, 1),
            _conv_bn_relu(channels, channels, 5, stride=stride),
            _conv_bn_relu(channels, out_channels, 1, activate=False),
        )
        if stride != 1 or in_channels != out_channels:
            shortcut: list[nn.Module] = []
            if stride != 1:
                shortcut.append(
                    nn.AvgPool1d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    )
                )
            if in_channels != out_channels:
                shortcut.extend(
                    [
                        nn.Conv1d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm1d(out_channels),
                    ]
                )
            self.shortcut = nn.Sequential(*shortcut)
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.shortcut(inputs))


class XResNet1d101(nn.Module):
    """xResNet1d101 at the architecture scale used by Strodthoff et al."""

    def __init__(
        self,
        num_outputs: int = 16,
        input_channels: int = 12,
        stage_blocks: Sequence[int] = (3, 4, 23, 3),
        base_width: int = 64,
        head_hidden: int = 128,
        head_dropout: tuple[float, float] = (0.25, 0.5),
    ) -> None:
        super().__init__()
        if tuple(stage_blocks) != (3, 4, 23, 3):
            raise ValueError("xResNet1d101 requires stage blocks (3, 4, 23, 3)")
        self.stem = nn.Sequential(
            _conv_bn_relu(input_channels, base_width // 2, 5, stride=2),
            _conv_bn_relu(base_width // 2, base_width // 2, 5),
            _conv_bn_relu(base_width // 2, base_width, 5),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        in_channels = base_width
        # The ECG benchmark deliberately keeps all four stages narrow. This is
        # the material architectural difference from an ImageNet ResNet-101.
        for stage_index, (channels, blocks) in enumerate(
            zip((base_width,) * 4, stage_blocks, strict=True)
        ):
            stage: list[nn.Module] = []
            for block_index in range(blocks):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stage.append(XResNetBottleneck1d(in_channels, channels, stride=stride))
                in_channels = channels * XResNetBottleneck1d.expansion
            stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*stages)
        pooled_channels = in_channels * 2
        self.head = nn.Sequential(
            nn.BatchNorm1d(pooled_channels),
            nn.Dropout(head_dropout[0]),
            nn.Linear(pooled_channels, head_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(head_hidden),
            nn.Dropout(head_dropout[1]),
            nn.Linear(head_hidden, num_outputs),
        )
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, XResNetBottleneck1d):
                nn.init.zeros_(module.body[-1][1].weight)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(inputs))
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=1)
        return self.head(pooled)


class S4DKernel(nn.Module):
    """Extension-free diagonal S4 convolution kernel.

    The discretization follows the official Apache-2.0 S4D formulation from
    ``state-spaces/s4``: diagonal continuous dynamics are discretized exactly and
    materialized as a convolution kernel. Separate learned readouts implement the
    two temporal directions.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        *,
        bidirectional: bool = True,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        if d_state < 2 or d_state % 2:
            raise ValueError("d_state must be a positive even real state dimension")
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        complex_modes = d_state // 2
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_a_real = nn.Parameter(torch.full((d_model, complex_modes), math.log(0.5)))
        a_imaginary = math.pi * torch.arange(complex_modes, dtype=torch.float32)
        self.register_buffer("a_imaginary", a_imaginary[None, :])
        directions = 2 if bidirectional else 1
        scale = 1 / math.sqrt(complex_modes)
        self.c_real = nn.Parameter(torch.randn(directions, d_model, complex_modes) * scale)
        self.c_imaginary = nn.Parameter(torch.randn(directions, d_model, complex_modes) * scale)

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if length < 1:
            raise ValueError("kernel length must be positive")
        dt = self.log_dt.exp()[:, None]
        a = torch.complex(-self.log_a_real.exp(), self.a_imaginary.expand_as(self.log_a_real))
        a_dt = a * dt
        a_bar = a_dt.exp()
        b_bar = torch.expm1(a_dt) / a
        powers = torch.arange(length, device=a.device, dtype=dt.dtype)
        vandermonde = a_bar[..., None] ** powers
        c = torch.complex(self.c_real, self.c_imaginary)
        kernels = 2 * torch.einsum("dhn,hnl,hn->dhl", c, vandermonde, b_bar).real
        backward = kernels[1] if self.bidirectional else None
        return kernels[0], backward


class S4DLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, bidirectional: bool = True) -> None:
        super().__init__()
        self.kernel = S4DKernel(d_model, d_state, bidirectional=bidirectional)
        self.skip = nn.Parameter(torch.randn(d_model))
        self._cached_frequency: torch.Tensor | None = None
        self._cached_length: int | None = None

    def train(self, mode: bool = True) -> S4DLayer:
        if mode:
            self._cached_frequency = None
            self._cached_length = None
        return super().train(mode)

    @staticmethod
    def _causal_convolution(inputs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[-1]
        fft_length = 2 * length
        inputs_frequency = torch.fft.rfft(inputs, n=fft_length)
        kernel_frequency = torch.fft.rfft(kernel, n=fft_length)
        return torch.fft.irfft(inputs_frequency * kernel_frequency[None], n=fft_length)[
            ..., :length
        ]

    @staticmethod
    def _bidirectional_convolution(
        inputs: torch.Tensor, forward_kernel: torch.Tensor, backward_kernel: torch.Tensor
    ) -> torch.Tensor:
        """Apply both directions with one zero-padded FFT convolution."""

        length = inputs.shape[-1]
        fft_length = 2 * length
        two_sided = F.pad(forward_kernel, (0, length))
        two_sided[:, 0] = two_sided[:, 0] + backward_kernel[:, 0]
        if length > 1:
            two_sided[:, -(length - 1) :] = backward_kernel[:, 1:].flip(-1)
        inputs_frequency = torch.fft.rfft(inputs, n=fft_length)
        kernel_frequency = torch.fft.rfft(two_sided, n=fft_length)
        return torch.fft.irfft(inputs_frequency * kernel_frequency[None], n=fft_length)[
            ..., :length
        ]

    @staticmethod
    def _two_sided_frequency(
        forward_kernel: torch.Tensor, backward_kernel: torch.Tensor
    ) -> torch.Tensor:
        length = forward_kernel.shape[-1]
        two_sided = F.pad(forward_kernel, (0, length))
        two_sided[:, 0] = two_sided[:, 0] + backward_kernel[:, 0]
        if length > 1:
            two_sided[:, -(length - 1) :] = backward_kernel[:, 1:].flip(-1)
        return torch.fft.rfft(two_sided, n=2 * length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[-1]
        if not self.training and not torch.is_grad_enabled() and self.kernel.bidirectional:
            if self._cached_frequency is None or self._cached_length != length:
                forward_kernel, backward_kernel = self.kernel(length)
                if backward_kernel is None:
                    raise RuntimeError("bidirectional kernel did not return its reverse direction")
                self._cached_frequency = self._two_sided_frequency(forward_kernel, backward_kernel)
                self._cached_length = length
            input_frequency = torch.fft.rfft(inputs, n=2 * length)
            output = torch.fft.irfft(input_frequency * self._cached_frequency[None], n=2 * length)[
                ..., :length
            ]
        else:
            forward_kernel, backward_kernel = self.kernel(length)
            if backward_kernel is None:
                output = self._causal_convolution(inputs, forward_kernel)
            else:
                output = self._bidirectional_convolution(inputs, forward_kernel, backward_kernel)
        return output + inputs * self.skip[None, :, None]


class S4DBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.sequence = S4DLayer(d_model, d_state, bidirectional=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = self.sequence(inputs)
        output = self.dropout(self.activation(output))
        output = F.glu(self.output(output), dim=1)
        output = self.dropout(output) + residual
        return self.normalization(output.transpose(1, 2)).transpose(1, 2)


class S4DClassifier(nn.Module):
    """Four-block bidirectional S4D classifier frozen in the protocol."""

    def __init__(
        self,
        num_outputs: int = 16,
        input_channels: int = 12,
        d_model: int = 512,
        d_state: int = 8,
        blocks: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.blocks = nn.Sequential(
            *(S4DBlock(d_model, d_state, dropout=dropout) for _ in range(blocks))
        )
        self.classifier = nn.Linear(d_model, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.encoder(inputs))
        return self.classifier(features.mean(dim=-1))


def build_model(name: str, num_outputs: int = 16) -> nn.Module:
    if name == "xresnet1d101":
        return XResNet1d101(num_outputs=num_outputs)
    if name == "s4d":
        return S4DClassifier(num_outputs=num_outputs)
    raise ValueError(f"unknown model: {name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
