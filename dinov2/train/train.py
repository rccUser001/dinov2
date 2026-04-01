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

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import (
    collate_data_and_cast, DataAugmentationDINO, DataAugmentationSlideflow,
    MaskingGenerator)
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def verify_dataloader_output(data, cfg):
    """One-time verification of dataloader output shapes and value ranges."""
    n_global_crops = 2
    n_local_crops = cfg.crops.local_crops_number
    batch_size = cfg.train.batch_size_per_gpu
    global_size = cfg.crops.global_crops_size
    local_size = cfg.crops.local_crops_size
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2

    logger.info("=" * 60)
    logger.info("DATALOADER VERIFICATION")
    logger.info("=" * 60)

    # Check global crops
    gc = data["collated_global_crops"]
    expected_gc_shape = (n_global_crops * batch_size, 3, global_size, global_size)
    logger.info(f"  global_crops shape: {tuple(gc.shape)} (expected: {expected_gc_shape})")
    if tuple(gc.shape) != expected_gc_shape:
        logger.warning(f"  MISMATCH in global_crops shape!")
    logger.info(f"  global_crops dtype: {gc.dtype}, range: [{gc.min().item():.4f}, {gc.max().item():.4f}]")
    if gc.min().item() == gc.max().item():
        logger.warning(f"  global_crops has ZERO variance — all values identical!")

    # Check local crops
    lc = data["collated_local_crops"]
    expected_lc_shape = (n_local_crops * batch_size, 3, local_size, local_size)
    logger.info(f"  local_crops shape: {tuple(lc.shape)} (expected: {expected_lc_shape})")
    if tuple(lc.shape) != expected_lc_shape:
        logger.warning(f"  MISMATCH in local_crops shape!")
    logger.info(f"  local_crops dtype: {lc.dtype}, range: [{lc.min().item():.4f}, {lc.max().item():.4f}]")

    # Check masks
    masks = data["collated_masks"]
    logger.info(f"  masks shape: {tuple(masks.shape)}, dtype: {masks.dtype}")
    logger.info(f"  masks — fraction masked: {masks.float().mean().item():.4f}")

    # Check mask indices
    mask_indices = data["mask_indices_list"]
    logger.info(f"  mask_indices_list shape: {tuple(mask_indices.shape)}, dtype: {mask_indices.dtype}")
    max_valid_index = n_global_crops * batch_size * n_tokens - 1
    if mask_indices.numel() > 0 and mask_indices.max().item() > max_valid_index:
        logger.warning(f"  mask_indices max {mask_indices.max().item()} exceeds valid range {max_valid_index}!")

    # Check for NaN/Inf
    for key in ["collated_global_crops", "collated_local_crops"]:
        t = data[key]
        if torch.isnan(t).any():
            logger.warning(f"  {key} contains NaN values!")
        if torch.isinf(t).any():
            logger.warning(f"  {key} contains Inf values!")

    logger.info("DATALOADER VERIFICATION COMPLETE")
    logger.info("=" * 60)


def log_gradient_stats(model, iteration):
    """Log gradient norms for positional embeddings and key parameters."""
    logger.info(f"[Iter {iteration}] GRADIENT FLOW CHECK:")

    # Check positional embeddings specifically (student only; teacher is frozen via EMA)
    found_pos_embed = False
    for name, param in model.named_parameters():
        if "pos_embed" in name:
            if name.startswith("teacher."):
                continue
            found_pos_embed = True
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.data.norm().item()
                logger.info(f"  {name}: grad_norm={grad_norm:.6f}, param_norm={param_norm:.6f}, "
                            f"grad/param_ratio={grad_norm / (param_norm + 1e-8):.6f}")
            else:
                logger.warning(f"  {name}: NO GRADIENT (None)")

    if not found_pos_embed:
        logger.warning("  No pos_embed parameters found in student model!")

    # Check a summary of all parameter groups
    total_params = 0
    zero_grad_params = 0
    no_grad_params = 0
    grad_norms = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total_params += 1
        if param.grad is None:
            no_grad_params += 1
        elif param.grad.norm().item() == 0.0:
            zero_grad_params += 1

        # Log norms per module group (backbone, dino_head, ibot_head)
        group = name.split(".")[1] if "." in name else name
        if group not in grad_norms:
            grad_norms[group] = []
        if param.grad is not None:
            grad_norms[group].append(param.grad.norm().item())

    for group, norms in grad_norms.items():
        if norms:
            avg_norm = sum(norms) / len(norms)
            max_norm = max(norms)
            logger.info(f"  {group}: avg_grad_norm={avg_norm:.6f}, max_grad_norm={max_norm:.6f}, "
                        f"num_params={len(norms)}")

    logger.info(f"  Summary: {total_params} trainable params, {no_grad_params} with no grad, "
                f"{zero_grad_params} with zero grad")


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
        "--local-rank",
        default=0,
        type=int,
        help="Variable for distributed computing."
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


def do_train(cfg, model, resume=False):
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

    checkpoint_period = cfg.train.get("saveckp_freq", 3 * OFFICIAL_EPOCH_LENGTH)
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=checkpoint_period,
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

    is_webdataset = cfg.train.dataset_path == "webdataset"
    is_slideflow = 'slideflow' in cfg.train and not is_webdataset
    aug_kw = dict(
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        convert_dtype=is_slideflow,
    )
    if is_slideflow:
        aug_class = DataAugmentationSlideflow
        if 'normalizer' in cfg.train.slideflow and cfg.train.slideflow.normalizer:
            aug_kw['normalizer'] = cfg.train.slideflow.normalizer
            logger.info("Using slideflow data augmentation with normalizer: {}".format(
                aug_kw['normalizer']
            ))
    else:
        aug_class = DataAugmentationDINO

    data_transform = aug_class(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        **aug_kw
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

    if is_webdataset:
        from dinov2.data.wds_loader import make_webdataset
        dataset = make_webdataset(cfg_train=cfg.train, image_transform=data_transform)
        sampler_type = None
    else:
        dataset = make_dataset(
            dataset_str=cfg.train.dataset_path,
            transform=data_transform,
            target_transform=lambda _: (),
            slideflow_args=(None if 'slideflow' not in cfg.train else cfg.train.slideflow)
        )
        # sampler_type = SamplerType.INFINITE
        sampler_type = SamplerType.SHARDED_INFINITE
    prefetch_factor = cfg.train.get("prefetch_factor", 2)
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
        persistent_workers=cfg.train.num_workers > 0,
        prefetch_factor=prefetch_factor,
    )

    # training loop

    iteration = start_iter
    total_tiles_seen = start_iter * cfg.train.batch_size_per_gpu * distributed.get_global_size()

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    dataloader_verified = False

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # One-time dataloader verification on first batch
        if not dataloader_verified:
            verify_dataloader_output(data, cfg)
            dataloader_verified = True

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

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

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        # Gradient flow check every 100 iterations (before optimizer.zero_grad next iter)
        if iteration % 100 == 0:
            log_gradient_stats(model, iteration)

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # Log individual loss components every 10 iterations for debugging
        if iteration % 10 == 0:
            components = "  ".join(f"{k}={v:.6f}" for k, v in loss_dict_reduced.items())
            logger.info(f"[Iter {iteration}] loss_components: total={losses_reduced:.6f}  {components}")

        total_tiles_seen += int(current_batch_size) * distributed.get_global_size()

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_tiles_seen=total_tiles_seen)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        if torch.cuda.is_available():
            metric_logger.update(
                gpu_mem_gb=torch.cuda.memory_allocated() / (1024 ** 3),
                gpu_max_mem_gb=torch.cuda.max_memory_allocated() / (1024 ** 3),
            )

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

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
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
