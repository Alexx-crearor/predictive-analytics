
from dataclasses import dataclass
import math

import numpy as np
import torch

from .integrated_gradients import target_prediction


@dataclass
class FaithfulnessResult:
    top_delta: torch.Tensor
    bottom_delta: torch.Tensor
    random_delta: torch.Tensor
    top_unit_ids: torch.Tensor
    bottom_unit_ids: torch.Tensor
    block_scores: torch.Tensor
    units_masked: int
    total_units: int


def score_blocks(ig, block_size=10):
    """(B,L,C) -> (B,n_blocks,C)."""
    if ig.ndim != 3 or block_size < 1:
        raise ValueError("IG shape hoặc block_size không hợp lệ")

    scores = []
    for start in range(0, ig.shape[1], block_size):
        block = ig[:, start:start + block_size, :]
        scores.append(block.abs().sum(dim=1))

    return torch.stack(scores, dim=1)


def mask_units(inputs, baseline, unit_ids, block_size):
    """Che k block-channel được chỉ định cho từng sample."""
    base = torch.broadcast_to(
        baseline.to(device=inputs.device, dtype=inputs.dtype),
        inputs.shape
    )

    changed = inputs.clone()
    channels = inputs.shape[2]

    for sample in range(inputs.shape[0]):
        for unit in unit_ids[sample].tolist():
            block, channel = divmod(int(unit), channels)
            start = block * block_size
            end = min(start + block_size, inputs.shape[1])

            changed[sample, start:end, channel] = (
                base[sample, start:end, channel]
            )

    return changed


@torch.no_grad()
def faithfulness_test(
    model,
    inputs,
    baseline,
    ig_attribution,
    *,
    horizon,
    target_channel=0,
    target_step=-1,
    block_size=10,
    top_fraction=0.2,
    random_repeats=10,
    seed=7,
):
    if model.training:
        raise ValueError("Cần gọi model.eval()")

    if inputs.ndim != 3 or inputs.shape != ig_attribution.shape:
        raise ValueError("Input và IG phải cùng shape (B,L,C)")

    if not 0 < top_fraction < 0.5:
        raise ValueError("top_fraction phải nằm trong (0,0.5)")

    if random_repeats < 1 or block_size < 1:
        raise ValueError("random_repeats và block_size phải dương")

    # 1. Xếp hạng các block theo độ lớn attribution
    scores = score_blocks(ig_attribution, block_size)
    flat_scores = scores.reshape(inputs.shape[0], -1)

    total_units = flat_scores.shape[1]
    k = max(1, math.ceil(total_units * top_fraction))

    ranking = torch.argsort(
        flat_scores, dim=1, descending=True, stable=True
    )

    top_ids = ranking[:, :k]
    bottom_ids = ranking[:, -k:]

    # 2. Chọn đúng một đầu ra dự báo
    def predict(x):
        return target_prediction(
            model(x),
            channels=inputs.shape[2],
            horizon=horizon,
            target_channel=target_channel,
            target_step=target_step,
        )

    original = predict(inputs)

    # 3. Đo mức thay đổi dự báo khi che
    def perturbation_delta(ids):
        changed = mask_units(
            inputs, baseline, ids, block_size
        )
        return (original - predict(changed)).abs()

    top_delta = perturbation_delta(top_ids)
    bottom_delta = perturbation_delta(bottom_ids)

    # 4. Random baseline, cùng số block và seed
    rng = np.random.default_rng(seed)
    random_delta = torch.zeros_like(top_delta)

    for _ in range(random_repeats):
        random_ids = np.stack([
            rng.permutation(total_units)[:k]
            for _ in range(inputs.shape[0])
        ])

        random_ids = torch.as_tensor(
            random_ids,
            dtype=torch.long,
            device=inputs.device,
        )

        random_delta += perturbation_delta(random_ids)

    random_delta /= random_repeats

    return FaithfulnessResult(
        top_delta=top_delta.cpu(),
        bottom_delta=bottom_delta.cpu(),
        random_delta=random_delta.cpu(),
        top_unit_ids=top_ids.cpu(),
        bottom_unit_ids=bottom_ids.cpu(),
        block_scores=scores.cpu(),
        units_masked=k,
        total_units=total_units,
    )