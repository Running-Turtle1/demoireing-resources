from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls, device_specification: str = "auto") -> DistributedContext:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if world_size > 1:
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = torch.device("cuda", local_rank)
                backend = "nccl"
            else:
                device = torch.device("cpu")
                backend = "gloo"
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    device_id=device if device.type == "cuda" else None,
                )
            except TypeError:
                dist.init_process_group(backend=backend, init_method="env://")
            return cls(rank, local_rank, world_size, device)

        device = _resolve_single_process_device(device_specification)
        return cls(rank=0, local_rank=0, world_size=1, device=device)

    def barrier(self) -> None:
        if self.enabled:
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


class DistributedEvalSampler(Sampler[int]):
    """Partition validation indices across ranks without padding or duplicates."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        num_replicas: int,
        rank: int,
        max_samples: int | None = None,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        sample_count = len(dataset)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive or None")
            sample_count = min(sample_count, max_samples)
        self.sample_count = sample_count
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, self.sample_count, self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= self.sample_count:
            return 0
        return (self.sample_count - 1 - self.rank) // self.num_replicas + 1


def reduce_sums(values: list[float], device: torch.device) -> list[float]:
    if not dist.is_available() or not dist.is_initialized():
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def reduce_max(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def gather_objects(value: Any, world_size: int) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    values: list[Any] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def _resolve_single_process_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device

