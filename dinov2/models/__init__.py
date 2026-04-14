# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    # `pretrained_weights_img_size`, if set, sizes the pos_embed parameter to
    # match the checkpoint's native training resolution (e.g. 518 for Meta's
    # ViT-L/14 DINOv2). The model's interpolate_pos_encoding() then downsamples
    # on the fly to the actual training crop size at each forward pass. This
    # preserves the full pretrained spatial grid instead of throwing it away
    # via strict=False shape-mismatch skipping.
    img_size = cfg.student.get("pretrained_weights_img_size", None)
    if img_size is None:
        img_size = cfg.crops.global_crops_size
    return build_model(cfg.student, only_teacher=only_teacher, img_size=img_size)
