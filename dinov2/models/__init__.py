# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits

from dinov2.models.block_expansion import expand_dinov2
from dinov2.models.lora_adaptation import apply_lora_to_dinov2


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224, cfg=None, expand=False, lora=False):
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
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim

    if expand:
        if cfg.block_expansion.enabled:
            student = expand_dinov2(student, cfg.block_expansion.expanded_blocks, cfg.block_expansion.path_dropout)
            teacher = expand_dinov2(teacher, cfg.block_expansion.expanded_blocks, cfg.block_expansion.path_dropout)
    
    if lora and cfg and hasattr(cfg, 'lora_adaptation') and cfg.lora_adaptation.enabled:
        student = apply_lora_to_dinov2(
            student,
            target_blocks=cfg.lora_adaptation.target_blocks,
            rank=cfg.lora_adaptation.rank,
            alpha=cfg.lora_adaptation.alpha,
            dropout=cfg.lora_adaptation.dropout,
            adapt_attention=cfg.lora_adaptation.adapt_attention,
            adapt_mlp=cfg.lora_adaptation.adapt_mlp,
        )
        teacher = apply_lora_to_dinov2(
            teacher,
            target_blocks=cfg.lora_adaptation.target_blocks,
            rank=cfg.lora_adaptation.rank,
            alpha=cfg.lora_adaptation.alpha,
            dropout=cfg.lora_adaptation.dropout,
            adapt_attention=cfg.lora_adaptation.adapt_attention,
            adapt_mlp=cfg.lora_adaptation.adapt_mlp,
        )
    
    if only_teacher:
        return teacher, teacher.embed_dim

    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False, expand=False, lora=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size, cfg=cfg, expand=expand, lora=lora)
