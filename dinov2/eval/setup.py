# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
import dinov2.utils.utils as dinov2_utils


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default=None,
        help="LoRA adapter weights to load (optional)",
    )
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Enable LoRA adaptation for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights, enable_lora=False, lora_weights=None):
    # Determine if LoRA should be enabled based on config and arguments
    use_lora = config.lora_adaptation.enabled and enable_lora
    
    model, _ = build_model_from_cfg(config, only_teacher=True, expand=True, lora=use_lora)
    
    # Load pretrained weights if provided
    if pretrained_weights:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    
    # Load LoRA weights if provided
    if use_lora and lora_weights:
        import torch
        from dinov2.models.lora_adaptation import load_lora_state_dict
        print(f"Loading LoRA weights from: {lora_weights}")
        lora_state_dict = torch.load(lora_weights, map_location="cpu")
        load_lora_state_dict(model, lora_state_dict)
    
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    model = build_model_for_eval(
        config, 
        args.pretrained_weights, 
        enable_lora=args.enable_lora, 
        lora_weights=getattr(args, 'lora_weights', None)
    )
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype
