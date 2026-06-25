# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Dependency-free LoRA integration for DINOv2 SSL pre-training.

This module is shipped with the Whole-Mouse-World-Model training stack and is
copied verbatim into the DINOv2 fork as ``dinov2/train/lora.py`` so it can be
imported as::

    from dinov2.train.lora import maybe_apply_lora_to_ssl_meta_arch

It has **no third-party dependencies beyond PyTorch** (OmegaConf is touched only
through duck-typed ``.get`` access, never imported), so the file is portable
between the WMWM repo and any DINOv2 checkout.

What it does
------------
``maybe_apply_lora_to_ssl_meta_arch(model, cfg)`` is a no-op unless the config
carries an enabled ``lora:`` block.  When enabled it:

  * wraps every targeted ``nn.Linear`` in **both** the student and teacher
    backbones with a low-rank adapter (:class:`LoRALinear`).  The original
    weight is preserved untouched as ``<name>.base.weight`` and two new
    parameters ``<name>.lora_A`` / ``<name>.lora_B`` are added;
  * freezes the student backbone's base weights (and every other backbone
    parameter -- pos-embed, patch-embed, norms, ...), leaving only the adapters
    trainable.  The DINO/iBOT heads outside the backbone stay fully trainable;
  * freezes the **whole** teacher (it is an EMA target, never optimized).  Its
    adapters still receive the EMA momentum update from the student;
  * **zero-initializes** ``lora_B`` so the adapter output is exactly zero at
    step 0 -- the model is numerically identical to the pretrained backbone
    (e.g. UNI) on the first forward pass.

Why the teacher also gets adapters (§ EMA)
------------------------------------------
DINOv2's ``update_teacher`` zips ``student.parameters()`` with
``teacher.parameters()`` positionally.  If only the student grew adapters the
two parameter lists would misalign and the EMA update would crash / corrupt.
Injecting the *same* structure into both keeps the lists aligned; the teacher's
adapters are frozen but ride the momentum update.

§7 Inference / merging
----------------------
A LoRA teacher checkpoint stores split keys (``...qkv.base.weight``,
``...qkv.lora_A``, ``...qkv.lora_B``).  Before running inference with the
unmodified ``load_teacher`` path, fold the adapters back into the base weights
with :func:`merge_lora_state_dict` (exposed via ``merge_lora_checkpoint.py``)::

    python scripts/train_dinov2/merge_lora_checkpoint.py \\
        --input teacher_checkpoint.pth --output teacher_merged.pth \\
        --config config.yaml

After merging the state dict has the original ``...qkv.weight`` layout again and
loads into a vanilla ViT unchanged.
"""

import logging
import math

import torch
from torch import nn

logger = logging.getLogger("dinov2")


# Default ``nn.Linear`` child names to wrap with an adapter. ``qkv`` is the
# fused query/key/value projection inside each attention block -- adapting it is
# the standard, parameter-cheap choice for ViT LoRA fine-tuning.
DEFAULT_TARGET_MODULES = ("qkv",)


# ---------------------------------------------------------------------------
# Small config helpers (work with OmegaConf DictConfig, plain dict, or argparse
# Namespace -- all expose either ``.get`` / ``__getitem__`` or attributes).
# ---------------------------------------------------------------------------
def _get(node, key, default=None):
    """Fetch ``key`` from a config-like ``node`` with a fallback."""
    if node is None:
        return default
    val = default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            val = node.get(key, default)
        except Exception:
            val = default
    else:
        try:
            val = node[key]
        except Exception:
            val = getattr(node, key, default)
    return default if val is None else val


def _first(node, keys, default=None):
    """Return the first present value among ``keys`` (alias resolution)."""
    sentinel = object()
    for key in keys:
        val = _get(node, key, sentinel)
        if val is not sentinel:
            return val
    return default


def _as_bool(val):
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(val)


def _as_list(val):
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    return list(val)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear``.

    Computes ``base(x) + scaling * (x @ A^T) @ B^T`` where ``A`` is
    ``[rank, in_features]`` and ``B`` is ``[out_features, rank]``.  ``B`` is
    zero-initialized so the adapter contributes nothing until trained, and the
    effective weight delta ``scaling * (B @ A)`` has the same ``[out, in]``
    shape as ``base.weight`` -- making merging a plain element-wise add.

    Naming is deliberate: the wrapped layer keeps its original parameters under
    ``self.base`` (``base.weight`` / ``base.bias``) and adds ``lora_A`` /
    ``lora_B``, so a checkpoint exposes ``<prefix>.base.weight`` /
    ``<prefix>.lora_A`` / ``<prefix>.lora_B``.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        # Create adapter params on the same device/dtype as the base weight so
        # this works whether or not the model has already been ``.to(cuda)``'d.
        weight = base.weight
        self.lora_A = nn.Parameter(weight.new_zeros((self.rank, self.in_features)))
        self.lora_B = nn.Parameter(weight.new_zeros((self.out_features, self.rank)))
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.reset_lora_parameters()

        # The base projection is frozen; only the adapters train.
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def reset_lora_parameters(self):
        # Kaiming-uniform A (standard LoRA init) and zero B -> zero delta at
        # init, but A carries signal so gradients flow once B leaves zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = self.base(x)
        lora = nn.functional.linear(self.dropout(x), self.lora_A)
        lora = nn.functional.linear(lora, self.lora_B)
        return out + self.scaling * lora

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Return ``base.weight + scaling * (B @ A)`` for checkpoint merging."""
        delta = self.scaling * (self.lora_B @ self.lora_A)
        return self.base.weight + delta.to(self.base.weight.dtype)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4g}"


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------
def _inject_adapters(module: nn.Module, target_modules, rank, alpha, dropout):
    """Recursively replace targeted ``nn.Linear`` children with ``LoRALinear``.

    Returns the number of layers wrapped. Idempotent: already-wrapped layers
    are skipped.
    """
    count = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and child_name in target_modules:
            setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            count += 1
        else:
            count += _inject_adapters(child, target_modules, rank, alpha, dropout)
    return count


def _freeze_backbone_base(backbone: nn.Module):
    """Freeze everything in ``backbone`` except the LoRA adapters."""
    for name, param in backbone.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def _count(params_iter):
    total, trainable = 0, 0
    for p in params_iter:
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def maybe_apply_lora_to_ssl_meta_arch(model, cfg):
    """Inject LoRA adapters into an ``SSLMetaArch`` if ``cfg.lora.enabled``.

    Call this in ``train.main()`` *after* the model is constructed (and the
    pretrained backbone is loaded) but *before*
    ``model.prepare_for_distributed_training()`` / FSDP wrapping.

    No-op when the ``lora`` config block is absent or ``enabled`` is false, so
    it is always safe to call unconditionally.
    """
    lora_cfg = _get(cfg, "lora", None)
    if lora_cfg is None or not _as_bool(_get(lora_cfg, "enabled", False)):
        return model

    rank = int(_first(lora_cfg, ("rank", "r"), 8))
    alpha = float(_first(lora_cfg, ("alpha", "lora_alpha"), rank))
    dropout = float(_first(lora_cfg, ("dropout", "lora_dropout"), 0.0))
    targets = _as_list(_first(lora_cfg, ("targets", "target_modules"), list(DEFAULT_TARGET_MODULES)))
    if not targets:
        targets = list(DEFAULT_TARGET_MODULES)
    targets = tuple(targets)

    student_backbone = model.student["backbone"]
    teacher_backbone = model.teacher["backbone"]

    # Inject the *same* adapter structure into both backbones so the student /
    # teacher parameter lists stay positionally aligned for the EMA update.
    n_student = _inject_adapters(student_backbone, targets, rank, alpha, dropout)
    n_teacher = _inject_adapters(teacher_backbone, targets, rank, alpha, dropout)

    # Student: only the adapters train inside the backbone; SSL heads (outside
    # the backbone) are left untouched and stay trainable.
    _freeze_backbone_base(student_backbone)

    # Teacher: EMA target -- freeze everything, including its fresh adapters.
    for param in model.teacher.parameters():
        param.requires_grad = False

    bb_total, bb_trainable = _count(student_backbone.parameters())
    model_total, model_trainable = _count(model.student.parameters())
    pct_bb = 100.0 * bb_trainable / max(bb_total, 1)
    pct_model = 100.0 * model_trainable / max(model_total, 1)

    logger.info(
        f"[LoRA] enabled rank={rank} alpha={alpha} dropout={dropout} "
        f"targets={list(targets)} | wrapped {n_student} student / {n_teacher} teacher "
        f"linear layers"
    )
    logger.info(
        f"[LoRA] trainable backbone params: {bb_trainable:,}/{bb_total:,} ({pct_bb:.3f}%) | "
        f"trainable student params (incl. heads): {model_trainable:,}/{model_total:,} ({pct_model:.3f}%)"
    )
    if n_student != n_teacher:
        logger.warning(
            f"[LoRA] student/teacher adapter count mismatch ({n_student} vs {n_teacher}); " "EMA pairing may misalign."
        )

    return model


# ---------------------------------------------------------------------------
# §7 checkpoint merging (imported by merge_lora_checkpoint.py)
# ---------------------------------------------------------------------------
def merge_lora_state_dict(state_dict, rank, alpha, target_modules=DEFAULT_TARGET_MODULES):
    """Fold LoRA adapters in a state dict back into their base weights.

    Converts split keys ``<prefix>.base.weight`` + ``<prefix>.lora_A`` +
    ``<prefix>.lora_B`` into a single ``<prefix>.weight = base + scaling*(B@A)``
    (and renames ``<prefix>.base.bias`` -> ``<prefix>.bias``).  Keys without
    adapters pass through unchanged.  Returns a new ``OrderedDict``.
    """
    from collections import OrderedDict

    scaling = float(alpha) / float(rank)
    merged = OrderedDict()
    # Group adapter tensors by their LoRALinear prefix (the part before ".base"
    # / ".lora_A" / ".lora_B").
    prefixes = set()
    for key in state_dict:
        for suffix in (".base.weight", ".lora_A", ".lora_B"):
            if key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])

    consumed = set()
    for prefix in prefixes:
        base_w_key = f"{prefix}.base.weight"
        a_key = f"{prefix}.lora_A"
        b_key = f"{prefix}.lora_B"
        if base_w_key not in state_dict or a_key not in state_dict or b_key not in state_dict:
            continue
        base_w = state_dict[base_w_key]
        lora_a = state_dict[a_key]
        lora_b = state_dict[b_key]
        delta = scaling * (lora_b @ lora_a)
        merged[f"{prefix}.weight"] = base_w + delta.to(base_w.dtype)
        consumed.update({base_w_key, a_key, b_key})

        base_b_key = f"{prefix}.base.bias"
        if base_b_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[base_b_key]
            consumed.add(base_b_key)

    for key, val in state_dict.items():
        if key not in consumed:
            merged[key] = val
    return merged
