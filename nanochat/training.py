"""
Shared training utilities for nanochat training scripts.

Extracts common patterns from base_train.py, chat_sft.py, and chat_rl.py
to reduce code duplication and ensure consistency across training stages.
"""

import gc
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized


class TrainingContext:
    """
    Encapsulates the common compute initialization boilerplate shared across
    all training and evaluation scripts (base_train, chat_sft, chat_rl, chat_eval, etc.).

    Usage:
        ctx = TrainingContext(device_type=args.device_type)
        # ctx.device, ctx.ddp_rank, ctx.master_process, etc. are all available
    """

    def __init__(self, device_type=""):
        device_type = autodetect_device_type() if device_type == "" else device_type
        self.device_type = device_type
        self.ddp, self.ddp_rank, self.ddp_local_rank, self.ddp_world_size, self.device = compute_init(device_type)
        self.master_process = self.ddp_rank == 0

        # Device-specific helpers
        if device_type == "cuda":
            self.synchronize = torch.cuda.synchronize
            self.get_max_memory = torch.cuda.max_memory_allocated
            self.gpu_device_name = torch.cuda.get_device_name(0)
            self.gpu_peak_flops = get_peak_flops(self.gpu_device_name)
            print0(f"GPU: {self.gpu_device_name} | Peak FLOPS (BF16): {self.gpu_peak_flops:.2e}")
        else:
            self.synchronize = lambda: None
            self.get_max_memory = lambda: 0
            self.gpu_device_name = None
            self.gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

        print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    def cleanup(self):
        compute_cleanup()


def init_wandb(run_name, project, master_process, user_config):
    """
    Initialize wandb or return a DummyWandb instance.

    Args:
        run_name: wandb run name ('dummy' disables logging)
        project: wandb project name
        master_process: whether this is the master process
        user_config: config dict to log to wandb
    """
    import wandb
    use_dummy = run_name == "dummy" or not master_process
    if use_dummy:
        return DummyWandb()
    return wandb.init(project=project, name=run_name, config=user_config)


def training_step(model, optimizer, train_loader, grad_accum_steps, scaler=None, next_fn=None):
    """
    Execute a single training step with gradient accumulation and optional GradScaler.

    Args:
        model: the compiled model
        optimizer: the optimizer
        train_loader: iterator yielding (inputs, targets, ...) batches
        grad_accum_steps: number of micro-steps for gradient accumulation
        scaler: optional torch.amp.GradScaler for fp16 training
        next_fn: callable to get next batch from loader; if None, uses next(train_loader)

    Returns:
        train_loss: the loss from the last micro-step (detached)
        extra: list of any extra values returned by the loader beyond (x, y)
    """
    extra = []
    for micro_step in range(grad_accum_steps):
        batch = next(train_loader) if next_fn is None else next_fn()
        x, y = batch[0], batch[1]
        if len(batch) > 2:
            extra = list(batch[2:])
        loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    return train_loss, extra


def optimizer_step(optimizer, scaler=None):
    """
    Step the optimizer with optional GradScaler support and distributed inf synchronization.

    Args:
        optimizer: the optimizer to step
        scaler: optional torch.amp.GradScaler for fp16 training
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def update_lr_and_momentum(optimizer, lr_multiplier, muon_momentum=None, muon_weight_decay=None):
    """
    Update learning rates and optionally Muon momentum/weight_decay for all param groups.

    Args:
        optimizer: the optimizer whose param_groups to update
        lr_multiplier: multiplier to apply to each group's initial_lr
        muon_momentum: if set, update momentum for 'muon' groups
        muon_weight_decay: if set, update weight_decay for 'muon' groups
    """
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_multiplier
        if group.get('kind') == 'muon':
            if muon_momentum is not None:
                group["momentum"] = muon_momentum
            if muon_weight_decay is not None:
                group["weight_decay"] = muon_weight_decay


def manage_gc(step, collect_interval=5000):
    """
    Manage garbage collection for training loops.
    Call after each training step. Freezes objects after step 1 and
    periodically collects to prevent memory buildup in long runs.

    Args:
        step: current training step (1-indexed)
        collect_interval: how often to manually collect (default 5000)
    """
    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % collect_interval == 0:
        gc.collect()


def ema_loss(smooth_loss, new_loss, step, beta=0.9):
    """
    Compute exponential moving average of loss with debiasing.

    Args:
        smooth_loss: current EMA value
        new_loss: new loss value to incorporate
        step: current step (0-indexed for debiasing)
        beta: EMA decay factor (default 0.9)

    Returns:
        (updated_smooth_loss, debiased_smooth_loss)
    """
    smooth_loss = beta * smooth_loss + (1 - beta) * new_loss
    debiased = smooth_loss / (1 - beta ** (step + 1))
    return smooth_loss, debiased


def reduce_scalar(value, device, op=dist.ReduceOp.SUM):
    """
    All-reduce a scalar value across distributed ranks.

    Args:
        value: scalar value to reduce
        device: torch device
        op: reduce operation (default SUM)

    Returns:
        reduced scalar value
    """
    tensor = torch.tensor([value], dtype=torch.float, device=device)
    dist.all_reduce(tensor, op=op)
    return tensor.item()


def reduce_counters(num_passed, total, device, ddp):
    """
    All-reduce pass/total counters across ranks for evaluation.

    Args:
        num_passed: number of passed examples on this rank
        total: total examples on this rank
        device: torch device
        ddp: whether distributed training is active

    Returns:
        (num_passed, total) aggregated across all ranks
    """
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()
    return num_passed, total


def compute_mfu(num_flops_per_token, total_batch_size, dt, gpu_peak_flops, ddp_world_size):
    """
    Compute model FLOPS utilization (MFU) percentage.

    Args:
        num_flops_per_token: estimated FLOPs per token
        total_batch_size: total batch size in tokens
        dt: time for one training step in seconds
        gpu_peak_flops: peak GPU FLOPS (BF16)
        ddp_world_size: number of distributed ranks

    Returns:
        (tok_per_sec, flops_per_sec, mfu_percent)
    """
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    return tok_per_sec, flops_per_sec, mfu
