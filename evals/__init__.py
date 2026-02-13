"""
Evaluation benchmarks package for GPT-2 training.

Provides a shared evaluation loop and benchmark-specific modules:
- hellaswag: HellaSwag sentence completion (10,042 val examples)
- mmlu: MMLU multiple-choice knowledge (curated ~12 subjects)
- arc: ARC-Challenge science reasoning (~1,170 test examples)

Each module exposes an evaluate(model, device, device_type, ddp, ddp_rank, ddp_world_size)
function that returns a float accuracy, keeping the training loop clean.
"""

import torch
from torch.nn import functional as F


def get_most_likely_row(tokens, mask, logits):
    """
    Given tokens, mask, and logits, returns the index of the completion with the lowest
    average loss (i.e., the most likely completion under the model).

    Works for any number of candidate completions (rows in tokens/mask/logits).
    """
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # the completion with the lowest loss is the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def evaluate_benchmark(model, device, device_type, ddp, ddp_rank, ddp_world_size,
                       iterate_fn, render_fn, benchmark_name):
    """
    Shared DDP-aware evaluation loop for all completion-style benchmarks.

    Args:
        model: the model to evaluate (already on device, may be DDP-wrapped)
        device: torch device string (e.g. 'cuda:0')
        device_type: 'cuda' or 'cpu'
        ddp: bool, whether DDP is active
        ddp_rank: int, this process's global rank
        ddp_world_size: int, total number of processes
        iterate_fn: callable that yields examples (format depends on render_fn)
        render_fn: callable(example) -> (tokens, mask, label) tensors
        benchmark_name: string name for printing (e.g. "HellaSwag")

    Returns:
        accuracy (float) -- normalized accuracy across all examples
    """
    import torch.distributed as dist

    model.eval()
    num_correct = 0
    num_total = 0

    for i, example in enumerate(iterate_fn()):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        tokens, mask, label = render_fn(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct += int(pred_norm == label)

    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct = torch.tensor(num_correct, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct = num_correct.item()

    acc = num_correct / num_total if num_total > 0 else 0.0

    # Print on master process (rank 0)
    master_process = (ddp_rank == 0)
    if master_process:
        print(f"{benchmark_name} accuracy: {num_correct}/{num_total}={acc:.4f}")

    return acc
