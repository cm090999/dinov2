# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("dinov2")

def resize_pos_embed_if_needed(checkpoint_state_dict, model_state_dict):
    """
    Resize the positional embeddings in case they don't match.
    
    Args:
        checkpoint_state_dict: The state_dict of the pretrained checkpoint
        model_state_dict: The state_dict of the current model
        
    Returns:
        checkpoint_state_dict: The updated state_dict with potentially resized position embeddings
    """
    if "pos_embed" in checkpoint_state_dict and "pos_embed" in model_state_dict:
        pos_embed_checkpoint = checkpoint_state_dict["pos_embed"]
        pos_embed_model = model_state_dict["pos_embed"]
        
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            logger.info(f"Positional embeddings shape mismatch: {pos_embed_checkpoint.shape} vs {pos_embed_model.shape}")
            
            # Separate class and patch embeddings
            if pos_embed_checkpoint.shape[1] > pos_embed_model.shape[1]:
                # Usually the first token is the class token
                class_token_checkpoint = pos_embed_checkpoint[:, 0:1, :]
                patch_embed_checkpoint = pos_embed_checkpoint[:, 1:, :]
                
                class_token_model = pos_embed_model[:, 0:1, :]
                patch_embed_model = pos_embed_model[:, 1:, :]
                
                # Interpolate patch embeddings
                patch_height_checkpoint = patch_width_checkpoint = int(patch_embed_checkpoint.shape[1] ** 0.5)
                patch_height_model = patch_width_model = int(patch_embed_model.shape[1] ** 0.5)
                
                patch_embed_checkpoint = patch_embed_checkpoint.reshape(
                    1, patch_height_checkpoint, patch_width_checkpoint, pos_embed_checkpoint.shape[2]
                ).permute(0, 3, 1, 2)
                
                patch_embed_model_resized = torch.nn.functional.interpolate(
                    patch_embed_checkpoint,
                    size=(patch_height_model, patch_width_model),
                    mode='bicubic',
                    align_corners=False
                )
                
                patch_embed_model_resized = patch_embed_model_resized.permute(0, 2, 3, 1).flatten(1, 2)
                
                # Combine class token with resized patch embeddings
                pos_embed_model_resized = torch.cat((class_token_checkpoint, patch_embed_model_resized), dim=1)
                
                # Replace checkpoint value with resized version
                checkpoint_state_dict["pos_embed"] = pos_embed_model_resized
                logger.info(f"Positional embeddings successfully resized to {pos_embed_model_resized.shape}")
    
    return checkpoint_state_dict


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # resize pos_embed if needed
    state_dict = resize_pos_embed_if_needed(state_dict, model.state_dict())
    # load state dict
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
