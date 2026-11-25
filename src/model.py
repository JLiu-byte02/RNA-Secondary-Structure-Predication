"""src/model.py -- RiNALMo backbone wrapper + pair-structure heads.

Two backbones share one interface ``forward(tokens, padding_mask) -> (B, T, D)``:

* :class:`RiNALMoBackbone` -- lazy import of the ``rinalmo`` package; modes
  ``frozen`` / ``full`` / ``last_k``, explicit gradient checkpointing.
* :class:`StubBackbone` -- 2-layer transformer with the same 1280-d interface,
  so unit tests and a CPU smoke run exercise the *real* training code path on
  machines where ``rinalmo`` (and flash-attn) are not installed.

Pair heads (``embed_dim -> L x L logits``, always symmetric):

* :class:`OuterProductPairHead` -- official RiNALMo ``SecStructPredictionHead``
  with the same attribute names, so official Zenodo checkpoints map 1:1;
* :class:`Wide2DHead` -- configurable-depth/width 2D resnet with optional
  ``[S0, BPP, stack, distance, mask]`` input channels.  The output conv is
  zero-initialised and added to the base head logits, so at step 0 the wide head
  is *numerically identical* to the plain outer-product head (a controlled
  starting point for every thermo-fusion ablation).

:class:`PairStructureModel` glues them together, strips ``<cls>/<eos>`` (with
per-sample lengths, so padded batches are handled correctly) and can load an
official checkpoint via ``load_official_state_dict``.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import torch.utils.checkpoint
from torch import nn

PAD_IDX = 1  # RiNALMo Alphabet: special tokens are <cls>,<pad>,<eos>,<unk>,<mask>


class BackboneUnavailableError(RuntimeError):
    """Raised when ``rinalmo`` is requested but not importable/pinned."""


# ---------------------------------------------------------------------------
# rinalmo import plumbing
# ---------------------------------------------------------------------------
_RINALMO_SEARCH_PATHS: List[Path] = []


def register_rinalmo_path(path: str | Path) -> None:
    """Allow an explicit checkout of RiNALMo to be imported (dev/test only)."""
    resolved = Path(path).resolve()
    if resolved not in _RINALMO_SEARCH_PATHS:
        _RINALMO_SEARCH_PATHS.append(resolved)


def ensure_rinalmo(extra_paths: Iterable[str | Path] | None = None):
    """Import and return the ``rinalmo`` package, or raise a clear error.

    Search order: the installed package first (the server installs it pinned),
    then explicitly registered checkouts (``third_party/RiNALMo-main``).

    A partial/namespace ``rinalmo`` (e.g. an editable install whose source tree
    is missing) can import successfully while lacking submodules, so readiness is
    validated by importing the modules we actually use, not just the top-level
    package.
    """
    import importlib

    def _ready() -> bool:
        importlib.import_module("rinalmo.config")
        importlib.import_module("rinalmo.data.alphabet")
        return True

    try:
        import rinalmo  # type: ignore

        if _ready():
            return rinalmo
    except Exception:
        pass

    candidates = list(_RINALMO_SEARCH_PATHS)
    if extra_paths:
        candidates.extend(Path(p) for p in extra_paths)
    last_error: Exception | None = None
    for candidate in candidates:
        candidate = Path(candidate)
        if not (candidate / "rinalmo").is_dir():
            continue
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            rinalmo = importlib.import_module("rinalmo")
            if _ready():
                return rinalmo
        except Exception as error:  # keep the last real error for the message
            last_error = error
            continue
    raise BackboneUnavailableError(
        "The 'rinalmo' package is not importable/complete. On the GPU server activate the "
        "project environment (activate.sh) after `pip install -e RiNALMo`; locally pass "
        "backbone.type=stub or register third_party/RiNALMo-main. "
        f"Underlying import error: {last_error!r}"
    ) from last_error


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop a leading ``module.`` prefix (DataParallel/DDP checkpoints)."""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_state_dict_file(path: str | Path, map_location="cpu") -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: expected a state_dict, got {type(payload)!r}")
    return _strip_module_prefix(payload)


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------
class StubBackbone(nn.Module):
    """Tiny transformer with the same interface/width as RiNALMo.

    ``embed_dim=1280`` by default so the pair head (and therefore every tensor
    shape in the training loop) matches the real RiNALMo-Mega/Giga path.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        num_layers: int = 2,
        num_heads: int = 20,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        vocab_size: int = 22,
        pad_idx: int = PAD_IDX,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.embed_dim = int(embed_dim)
        self.pad_idx = int(pad_idx)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # One independent layer per entry: sharing a single module would alias all
        # parameters (named_parameters dedupes them) and silently shrink the model.
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(dim_feedforward or 4 * embed_dim),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @property
    def pad_tkn_idx(self) -> int:
        return self.pad_idx

    def set_mode(self, mode: str, last_k: int | None = None) -> None:
        """Same frozen/full/last_k contract as :class:`RiNALMoBackbone`.

        Exists so the gradual-unfreezing schedule can be unit-tested on CPU with
        the stub backbone instead of a 650M encoder.
        """
        if mode not in ("frozen", "full", "last_k"):
            raise ValueError(f"unknown fine-tuning mode {mode!r}")
        if last_k is not None:
            try:
                self.last_k = int(last_k)
            except AttributeError:  # pragma: no cover - defensive
                self.last_k = int(last_k)
        for parameter in self.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "last_k":
            layers = list(self.layers)
            k = int(getattr(self, "last_k", 1))
            if not 1 <= k <= len(layers):
                raise ValueError(f"last_k={k} out of range 1..{len(layers)}")
            for layer in layers[-k:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.final_layer_norm.parameters():
                parameter.requires_grad = True

    @property
    def trainable_parameter_names(self) -> List[str]:
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_idx)
        x = self.embedding(tokens)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, src_key_padding_mask=padding_mask, use_reentrant=False
                )
            else:
                x = layer(x, src_key_padding_mask=padding_mask)
        x = self.final_layer_norm(x)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class RiNALMoBackbone(nn.Module):
    """RiNALMo encoder with frozen / full / last-k fine-tuning modes.

    ``flash_attn``:
        ``"auto"`` uses flash-attn only if it is importable (the official giga
        config sets ``use_flash_attn=True``); ``True``/``False`` force the path.
    ``gradient_checkpointing``:
        Recomputation is applied here explicitly (RiNALMo's own ``Transformer``
        also does it unconditionally; we bypass that loop to control the mode
        and to skip the unused MLM head during fine-tuning).
    """

    def __init__(
        self,
        lm_config: str = "mega",
        *,
        pretrained_path: str | Path | None = None,
        model_name: str | None = None,
        mode: str = "full",
        last_k: int = 4,
        gradient_checkpointing: bool = True,
        flash_attn: bool | str = "auto",
        frozen_no_grad: bool = True,
        allow_random_init: bool = False,
        extra_paths: Iterable[str | Path] | None = None,
        load_weights: bool = True,
    ):
        super().__init__()
        if mode not in ("frozen", "full", "last_k"):
            raise ValueError(f"unknown fine-tuning mode {mode!r}")
        ensure_rinalmo(extra_paths)
        from rinalmo.config import model_config  # type: ignore
        from rinalmo.model.model import RiNALMo  # type: ignore

        config = model_config(lm_config)
        self._apply_flash_attn(config, flash_attn)
        model = RiNALMo(config)
        self.model = model
        self.lm_config = lm_config
        self.mode = mode
        self.last_k = int(last_k)
        self.frozen_no_grad = bool(frozen_no_grad)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.embed_dim = int(model.config.model.transformer.embed_dim)
        self.pad_idx = int(model.pad_tkn_idx)
        self.pretrained_path = str(pretrained_path) if pretrained_path else None

        self.load_report: Dict[str, Any] = {}
        if load_weights:
            if pretrained_path:
                state_dict = load_state_dict_file(pretrained_path)
                # Official fine-tuned checkpoints (Zenodo) wrap the encoder as
                # ``lm.*`` next to ``pred_head.*``/``threshold``; keep only the
                # encoder so it can initialise full fine-tuning as well as the
                # frozen protocol.
                if any(key.startswith("lm.") for key in state_dict):
                    state_dict = {
                        key[len("lm.") :]: value
                        for key, value in state_dict.items()
                        if key.startswith("lm.")
                    }
                report = model.load_state_dict(state_dict, strict=True)
                self.load_report = {
                    "source": str(pretrained_path),
                    "missing": list(report.missing_keys),
                    "unexpected": list(report.unexpected_keys),
                }
            elif model_name:
                from rinalmo.pretrained import get_pretrained_model  # type: ignore

                loaded, _alphabet = get_pretrained_model(model_name, lm_config=lm_config)
                self.model = model = loaded
                self.embed_dim = int(model.config.model.transformer.embed_dim)
                self.load_report = {"source": f"pretrained:{model_name}"}
            elif not allow_random_init:
                raise BackboneUnavailableError(
                    "RiNALMoBackbone needs pretrained_path or model_name (or "
                    "allow_random_init=True for architecture-only tests). "
                    "Pretrained weights must be an explicit local file on the server."
                )
        self.set_mode(mode, last_k=last_k)

    # -- construction helpers ---------------------------------------------
    @staticmethod
    def _apply_flash_attn(config, flash_attn: bool | str) -> None:
        if flash_attn == "auto":
            try:
                import flash_attn  # noqa: F401

                enabled = True
            except Exception:
                enabled = False
        else:
            enabled = bool(flash_attn)
        config.model.transformer.use_flash_attn = enabled

    def set_mode(self, mode: str, last_k: int | None = None) -> None:
        if mode not in ("frozen", "full", "last_k"):
            raise ValueError(f"unknown fine-tuning mode {mode!r}")
        self.mode = mode
        if last_k is not None:
            self.last_k = int(last_k)
        for parameter in self.model.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "last_k":
            blocks = list(self.model.transformer.blocks)
            if not 1 <= self.last_k <= len(blocks):
                raise ValueError(
                    f"last_k={self.last_k} out of range 1..{len(blocks)} for {self.lm_config}"
                )
            for block in blocks[-self.last_k :]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.model.transformer.final_layer_norm.parameters():
                parameter.requires_grad = True

    @property
    def trainable_parameter_names(self) -> List[str]:
        return [name for name, p in self.named_parameters() if p.requires_grad]

    # -- forward -----------------------------------------------------------
    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        model = self.model
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_idx)
        use_flash = bool(model.config.model.transformer.use_flash_attn)
        # RiNALMo's own convention is inverted for flash-attn kernels.
        key_padding_mask = torch.logical_not(padding_mask) if use_flash else padding_mask
        trainable = any(p.requires_grad for p in model.parameters())
        if self.mode == "frozen" and self.frozen_no_grad and not trainable:
            with torch.no_grad():
                return self._run_layers(tokens, key_padding_mask)
        return self._run_layers(tokens, key_padding_mask)

    def _run_layers(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        model = self.model
        x = model.embedding(tokens)
        x = model.token_dropout(x, tokens)
        checkpointing = self.gradient_checkpointing and self.training and x.requires_grad
        for block in model.transformer.blocks:
            if checkpointing:
                x, _ = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    key_padding_mask=key_padding_mask,
                    need_attn_weights=False,
                    use_reentrant=False,
                )
            else:
                x, _ = block(x, key_padding_mask=key_padding_mask, need_attn_weights=False)
        return model.transformer.final_layer_norm(x)


# ---------------------------------------------------------------------------
# Pair heads
# ---------------------------------------------------------------------------
def outer_concat(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Official ``_outer_concat``: ``(B, L, E) -> (B, L, L, 2E)``."""
    if left.shape != right.shape:
        raise ValueError(f"outer_concat shapes differ: {left.shape} vs {right.shape}")
    seq_len = left.shape[1]
    a = left.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = right.unsqueeze(-3).expand(-1, seq_len, -1, -1)
    return torch.cat((a, b), dim=-1)


class ResNet2DBlock(nn.Module):
    """Official ``ResNet2DBlock`` (1x1 -> k x k -> 1x1 bottleneck + residual)."""

    def __init__(self, embed_dim: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=kernel_size,
                bias=bias,
                padding="same",
            ),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_net(x)


class ResNet2D(nn.Module):
    def __init__(self, embed_dim: int, num_blocks: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            ResNet2DBlock(embed_dim, kernel_size=kernel_size, bias=bias)
            for _ in range(int(num_blocks))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def symmetrize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Official symmetry trick: keep the strict upper triangle, mirror it.

    The diagonal becomes exactly 0 (sigmoid 0.5); the decoder never looks at it.
    """
    upper = torch.triu(logits, diagonal=1)
    return upper + upper.transpose(-1, -2)


class OuterProductPairHead(nn.Module):
    """Official RiNALMo ``SecStructPredictionHead`` (num_blocks configurable).

    Attribute names are kept identical to the official class so that
    ``state_dict`` keys from Zenodo checkpoints (``pred_head.*``) line up.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 2,
        conv_dim: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.conv_dim = int(conv_dim)
        self.kernel_size = int(kernel_size)
        self.linear_in = nn.Linear(embed_dim * 2, conv_dim)
        self.resnet = ResNet2D(conv_dim, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # W[h_i; h_j] + b = W_left h_i + W_right h_j + b.
        # Project before broadcasting to avoid materialising an L x L x 2E tensor.
        left_weight, right_weight = self.linear_in.weight.split(self.embed_dim, dim=1)
        left = nn.functional.linear(emb, left_weight, self.linear_in.bias)
        right = nn.functional.linear(emb, right_weight)
        x = left.unsqueeze(-2) + right.unsqueeze(-3)
        x = x.permute(0, 3, 1, 2)  # B, C, L, L
        x = self.resnet(x)
        x = self.conv_out(x).squeeze(1)  # B, L, L
        return symmetrize_logits(x)


class Wide2DHead(nn.Module):
    """Configurable 2D resnet with optional physics/geometry input channels.

    Channel order (``self.channels``) is explicit and recorded in the checkpoint
    config.  ``s0`` is the base outer-product head logit (so the wide head is a
    strict generalisation of :class:`OuterProductPairHead`); ``bpp``/``stack``/
    ``mfe``/``mea``/``centroid`` come from the dataset, ``distance`` is rebuilt
    from the sequence length and ``mask`` from the batch pair mask.

    ``zero_channels`` keeps the channel in the architecture (so parameter count
    and optimisation are unchanged) but replaces its values with zeros at every
    forward pass.  This is the matched zero-BPP control: the network sees exactly
    the same inputs except the thermodynamic signal.

    The final 1x1 conv is zero-initialised: ``logits == base_logits`` at step 0.
    """

    ALL_CHANNELS = ("s0", "bpp", "stack", "mfe", "mea", "centroid", "distance", "mask")

    def __init__(
        self,
        embed_dim: int,
        *,
        channels: Sequence[str] = ("s0", "bpp", "stack", "distance", "mask"),
        conv_dim: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
        base_blocks: int = 2,
        base_conv_dim: int = 64,
        zero_init: bool = True,
        zero_channels: Sequence[str] = (),
    ):
        super().__init__()
        channels = tuple(channels)
        unknown = [c for c in channels if c not in self.ALL_CHANNELS]
        if unknown:
            raise ValueError(f"unknown head channels {unknown}; supported {self.ALL_CHANNELS}")
        if not channels:
            raise ValueError("Wide2DHead needs at least one input channel")
        zero_channels = tuple(zero_channels)
        unknown_zero = [c for c in zero_channels if c not in channels]
        if unknown_zero:
            raise ValueError(
                f"zero_channels {unknown_zero} must be a subset of head channels {channels}"
            )
        self.embed_dim = int(embed_dim)
        self.channels = channels
        self.zero_channels = zero_channels
        self.base = OuterProductPairHead(
            embed_dim, num_blocks=base_blocks, conv_dim=base_conv_dim, kernel_size=kernel_size
        )
        self.input_conv = nn.Conv2d(len(channels), conv_dim, kernel_size=1)
        self.resnet = ResNet2D(conv_dim, depth, kernel_size)
        self.out_conv = nn.Conv2d(conv_dim, 1, kernel_size=1)
        if zero_init:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

    def _channel_tensor(
        self,
        name: str,
        *,
        base_logits: torch.Tensor,
        features: Mapping[str, torch.Tensor] | None,
        pair_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length = base_logits.shape[0], base_logits.shape[1]
        if name in self.zero_channels:
            return base_logits.new_zeros((batch, 1, length, length))
        if name == "s0":
            return base_logits.unsqueeze(1)
        if name in ("bpp", "stack", "mfe", "mea", "centroid"):
            if features is None or name not in features:
                raise KeyError(
                    f"head channel {name!r} requested but batch carries no {name!r} feature; "
                    "set data.channels / model.channels consistently"
                )
            value = features[name]
            if value.dim() == 2:  # unbatched (L, L)
                value = value.unsqueeze(0)
            if value.shape[-2:] != (length, length):
                raise ValueError(
                    f"feature {name!r} has shape {tuple(value.shape)}, expected "
                    f"({batch}, {length}, {length})"
                )
            return value.unsqueeze(1)
        if name == "distance":
            positions = torch.arange(length, device=base_logits.device, dtype=torch.float32)
            distance = (positions[:, None] - positions[None, :]).abs()
            distance = distance / max(length - 1, 1)
            return distance.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)
        if name == "mask":
            if pair_mask is None:
                return base_logits.new_ones((batch, 1, length, length))
            mask = pair_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            return mask.unsqueeze(1).to(base_logits.dtype)
        raise ValueError(f"unreachable channel {name!r}")

    def forward(
        self,
        emb: torch.Tensor,
        features: Mapping[str, torch.Tensor] | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_logits = self.base(emb)
        channel_tensors = [
            self._channel_tensor(
                name, base_logits=base_logits, features=features, pair_mask=pair_mask
            )
            for name in self.channels
        ]
        x = torch.cat(channel_tensors, dim=1)
        x = self.input_conv(x)
        x = self.resnet(x)
        correction = self.out_conv(x).squeeze(1)
        logits = base_logits + correction
        return symmetrize_logits(logits)


def build_head(head_config: Mapping[str, Any] | None, embed_dim: int) -> nn.Module:
    """Build a pair head from config::

        {"type": "outer_product" | "wide2d", "kwargs": {...}}
    """
    head_config = dict(head_config or {})
    head_type = str(head_config.get("type", "outer_product")).lower()
    kwargs = dict(head_config.get("kwargs", {}))
    if head_type in ("outer_product", "official", "outer"):
        return OuterProductPairHead(embed_dim, **kwargs)
    if head_type in ("wide2d", "wide_2d", "resnet2d"):
        return Wide2DHead(embed_dim, **kwargs)
    raise ValueError(f"unknown head type {head_type!r}")


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class PairStructureModel(nn.Module):
    """Backbone + pair head; accepts padded token batches.

    ``forward`` returns ``(B, Lmax, Lmax)`` logits where positions beyond each
    RNA's length are meaningless (masked out by the loss / sliced by the
    decoder).
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    # -- plumbing ----------------------------------------------------------
    @property
    def embed_dim(self) -> int:
        return int(self.backbone.embed_dim)

    @property
    def head_channels(self) -> tuple[str, ...]:
        return tuple(getattr(self.head, "channels", ()))

    def data_channels(self) -> List[str]:
        """Dataset channels required by the head (``s0`` is internal)."""
        return [c for c in self.head_channels if c not in ("s0", "distance", "mask")]

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        channels: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        channel_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        representation = self.backbone(tokens, padding_mask=padding_mask)
        emb = self._sequence_embeddings(representation, tokens, lengths, padding_mask)
        features = None
        if channels is not None:
            names = list(channel_names if channel_names is not None else self.data_channels())
            features = {name: channels[:, index] for index, name in enumerate(names)}
        if isinstance(self.head, Wide2DHead):
            logits = self.head(emb, features=features, pair_mask=pair_mask)
        else:
            logits = self.head(emb)
        return logits

    @staticmethod
    def _sequence_embeddings(
        representation: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Drop ``<cls>``/``<eos>`` and padding -> ``(B, Lmax, D)``.

        The canonical layout is ``<cls> seq <eos> <pad>...``: the sequence always
        starts at index 1, so ``representation[:, 1:1+Lmax]`` is the sequence
        block followed by ``<eos>/<pad>``; positions ``>= length`` are zeroed.
        """
        batch, total_tokens, dim = representation.shape
        if lengths is None:
            if padding_mask is not None:
                lengths = (~padding_mask).sum(dim=1) - 2
            else:
                lengths = torch.full(
                    (batch,), total_tokens - 2, device=representation.device, dtype=torch.long
                )
        lengths = lengths.to(representation.device)
        max_len = int(lengths.max().item())
        if max_len > total_tokens - 2:
            raise ValueError(f"length {max_len} does not fit in {total_tokens} tokens")
        seq = representation[:, 1 : 1 + max_len, :]
        positions = torch.arange(max_len, device=representation.device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        return seq * valid.unsqueeze(-1).to(seq.dtype)

    # -- parameter utilities ----------------------------------------------
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }

    # -- official checkpoint interoperability ------------------------------
    def load_official_state_dict(
        self, source: str | Path | Mapping[str, torch.Tensor], strict: bool = False
    ) -> Dict[str, Any]:
        """Map ``lm.*`` / ``pred_head.*`` Zenodo keys onto this model.

        Returns a small report (mapped/unexpected/missing/ignored threshold) so a
        run can log exactly how much of the official checkpoint was reused.
        """
        if isinstance(source, (str, Path)):
            state_dict = load_state_dict_file(source)
        else:
            state_dict = _strip_module_prefix(source)
        threshold = state_dict.pop("threshold", None)
        mapped: Dict[str, torch.Tensor] = {}
        unexpected: List[str] = []
        for key, value in state_dict.items():
            if key.startswith("lm."):
                mapped[f"backbone.model.{key[len('lm.'):]}"] = value
            elif key.startswith("pred_head."):
                if isinstance(self.head, Wide2DHead):
                    mapped[f"head.base.{key[len('pred_head.'):]}"] = value
                else:
                    mapped[f"head.{key[len('pred_head.'):]}"] = value
            else:
                unexpected.append(key)
        missing, unexpected_model = self.load_state_dict(mapped, strict=False)
        return {
            "threshold": None if threshold is None else float(threshold),
            "mapped_keys": len(mapped),
            "missing_keys": list(missing),
            "unexpected_keys": sorted(unexpected + list(unexpected_model)),
            "strict": bool(strict),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_STUB_KEYS = (
    "embed_dim",
    "num_layers",
    "num_heads",
    "dim_feedforward",
    "dropout",
    "vocab_size",
    "pad_idx",
    "gradient_checkpointing",
)
_RINALMO_KEYS = (
    "lm_config",
    "pretrained_path",
    "model_name",
    "mode",
    "last_k",
    "gradient_checkpointing",
    "flash_attn",
    "frozen_no_grad",
    "allow_random_init",
    "extra_paths",
    "load_weights",
)


def build_backbone(backbone_config: Mapping[str, Any]) -> nn.Module:
    """Build a backbone, tolerating encoder-specific keys on the stub.

    CLI overrides such as ``--set model.backbone.type=stub`` leave the rest of
    the RiNALMo config in place (``lm_config``, ``pretrained_path``, ...); those
    keys are meaningless for :class:`StubBackbone` and must not crash a CPU smoke
    run.  Unknown keys for a real backbone still raise, so typos are caught.
    """
    config = dict(backbone_config)
    backbone_type = str(config.pop("type", "rinalmo")).lower()
    if backbone_type in ("stub", "stub_backbone"):
        filtered = {key: value for key, value in config.items() if key in _STUB_KEYS}
        return StubBackbone(**filtered)
    if backbone_type in ("rinalmo", "rinalmo_backbone"):
        unknown = set(config) - set(_RINALMO_KEYS)
        if unknown:
            raise ValueError(f"unknown rinalmo backbone config keys: {sorted(unknown)}")
        filtered = {key: value for key, value in config.items() if key in _RINALMO_KEYS}
        return RiNALMoBackbone(**filtered)
    raise ValueError(f"unknown backbone type {backbone_type!r}")


def build_model(config: Mapping[str, Any]) -> PairStructureModel:
    """Build from the ``model:`` section of a YAML config."""
    if "backbone" not in config:
        raise ValueError("model config must contain a 'backbone' section")
    backbone = build_backbone(config["backbone"])
    head = build_head(config.get("head"), embed_dim=backbone.embed_dim)
    return PairStructureModel(backbone, head)


def model_signature(model: nn.Module) -> str:
    """Stable hash of the architecture + parameter shapes (resume provenance)."""
    description = []
    for name, parameter in model.named_parameters():
        description.append(f"{name}:{tuple(parameter.shape)}")
    return hashlib.sha256("\n".join(description).encode("utf-8")).hexdigest()
