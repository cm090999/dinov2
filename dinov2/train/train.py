# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch
import aim

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch

from dinov2.eval.metrics import AccuracyAveraging

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")

def periodic_eval(model, cfg, iteration, step, aim_run=None):

    chkpt_path = do_test(cfg, model, f"training_iteration_{iteration}_step_{step}")

    ############  KNN EVAL  ############
    from dinov2.eval.knn import eval_knn
    from dinov2.eval.setup import build_model_for_eval
    from dinov2.data.transforms import make_classification_eval_transform
    import json

    # Multiple eval datasets can be specified by separating eval_dataset and eval_dataset_name with >
    dataset_list = cfg.evaluation.eval_dataset.split(">")
    dataset_name_list = cfg.evaluation.eval_dataset_name.split(">")

    for i, (dataset, dataset_name) in enumerate(zip(dataset_list, dataset_name_list)):

        eval_model = build_model_for_eval(cfg, chkpt_path, enable_lora=True)
        eval_transform = make_classification_eval_transform()

        eval_dataset = make_dataset(
            dataset_str=dataset,
            transform=eval_transform,
        )
        eval_train_size = int(len(eval_dataset) * cfg.evaluation.train_fraction)
        eval_val_size = len(eval_dataset) - eval_train_size
        eval_train_dataset, eval_val_dataset = torch.utils.data.random_split(
            eval_dataset, [eval_train_size, eval_val_size]
        )
        
        # Run eval
        results_dict_knn = eval_knn(eval_model,eval_train_dataset,eval_val_dataset,accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,nb_knn=(10, 20, 100, 200),temperature=0.07,batch_size=256,num_workers=48,gather_on_cpu=False,)
        metrics_file_path = os.path.join(cfg.train.output_dir, "eval", f"eval_results_iteration_{iteration}_step_{step}_{dataset_name}.csv")
        
        import pandas as pd
        knn_df = []
        if distributed.is_main_process():
            for neighbor in results_dict_knn.keys():
                for topx in results_dict_knn[neighbor].keys():
                    neighbor_name = neighbor[1]
                    log_name = f"knn_neighbors_{neighbor_name}_top_{topx}_{dataset_name}"
                    result_value = results_dict_knn[neighbor][topx].item()
                    #append row with neighbor, topx, and result_value
                    knn_df.append({"neighbor": neighbor_name, "topx": topx, "result_value": result_value})
                    if aim_run:
                        aim_run.track(result_value, name=f'eval/{log_name}', step=step, context={"subset": "eval"})

        if distributed.is_main_process():
            # Save results to CSV
            knn_df = pd.DataFrame(knn_df)
            knn_df.to_csv(metrics_file_path, index=False)
            logger.info(f"Saved KNN results to {metrics_file_path}")

        if distributed.is_enabled():
            torch.distributed.barrier()
    ############\ KNN EVAL \############

    torch.cuda.synchronize()


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        default="dinov2_experiment",
        type=str,
        help="Name of the experiment for logging purposes",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)

    return teacher_ckp_path


def do_train(cfg, model, resume=False, aim_run=None):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop
    accumulation_steps = cfg.train.accumulation_steps
    iteration = start_iter
    step = 0 # TODO: Not considering start_iter for now
    last_eval_step = 0
    last_checkpointer_step = 0

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file, aim_run=aim_run) # Pass aim_run to MetricLogger
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10 * accumulation_steps,
        header,
        max_iter * accumulation_steps,
        start_iter,
    ):
        metric_logger.current_step = step  # Set current_step for the logger
        
        # First iteration eval
        if cfg.evaluation.eval_period_iterations > 0 and iteration == 0:
            periodic_eval(model, cfg, iteration, step, aim_run=aim_run)
            last_eval_step = step

        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if step > max_iter:
            return

        # apply schedules
        lr = lr_schedule[step]
        wd = wd_schedule[step]
        mom = momentum_schedule[step]
        teacher_temp = teacher_temp_schedule[step]
        last_layer_lr = last_layer_lr_schedule[step]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses and accumulate gradients
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp, accumulation_steps=accumulation_steps)

        # clip gradients, optimizer step, scaler update, and teacher EMA update
        # are performed only every accumulation_steps
        if (iteration + 1) % accumulation_steps == 0:
            if fp16_scaler is not None:
                if cfg.optim.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                if cfg.optim.clip_grad:
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                optimizer.step()
            step += 1  # Increment step after successful optimizer step

            # perform teacher EMA update
            model.update_teacher(mom)

            # Zero gradients for the next accumulation cycle
            optimizer.zero_grad(set_to_none=True)

        # logging
        metric_logger.current_step = step  # Update step in logger before logging metrics for the iteration

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (step + 1) % cfg.evaluation.eval_period_iterations == 0:
            if last_eval_step == step:
                pass
            else:
                periodic_eval(model, cfg, iteration, step, aim_run=aim_run)
                last_eval_step = step

        if step > last_checkpointer_step:
            periodic_checkpointer.step(step)
            last_checkpointer_step = step

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    cfg = setup(args)

    # Initialize Aim run
    if distributed.is_main_process():
        if not os.path.exists(cfg.train.repo_dir):
            os.makedirs(cfg.train.repo_dir)
        aim_run = aim.Run(experiment=cfg.train.experiment_name,
                          repo=cfg.train.repo_dir,
                          )
        aim_run['hparams'] = cfg
    else:
        aim_run = None

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        periodic_eval(model, cfg, 0, 0, aim_run=aim_run)
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume, aim_run=aim_run)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
