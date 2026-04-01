import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Data Parallelism
# ---------------------------------------------------------------------------

class DataParallelModel(nn.Module):
    """Data Parallelism: same model replicated on each GPU, different data shards.

    How it works:
    - Each GPU holds a full copy of the model.
    - The batch is split across GPUs: each processes B/N samples.
    - Forward and backward passes run independently on each GPU.
    - Gradients are all-reduced (summed/averaged) across GPUs before the optimizer step.

    PyTorch built-ins:
        DDP  (DistributedDataParallel) — preferred, each process owns one GPU, gradient sync via all-reduce.
        FSDP (FullyShardedDataParallel) — shards model params + grads + optimizer states across GPUs,
              enabling models too large for a single GPU while using the same data-parallel training loop.

    This is a minimal manual illustration — use DDP/FSDP in practice.
    """

    def __init__(self, model: nn.Module, device_ids: List[int]):
        super().__init__()
        self.model = model
        self.device_ids = device_ids
        self.devices = [torch.device(f"cuda:{i}") for i in device_ids]
        self.replicas = [model.to(d) for d in self.devices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(len(self.devices), dim=0)                         # split batch
        outputs = [
            replica(chunk.to(device))
            for replica, chunk, device in zip(self.replicas, chunks, self.devices)
        ]
        return torch.cat([o.to(self.devices[0]) for o in outputs], dim=0)  # gather on device 0

    @staticmethod
    def sync_gradients(replicas: List[nn.Module]):
        """Average gradients across all replicas (manual all-reduce)."""
        n = len(replicas)
        for params in zip(*[r.parameters() for r in replicas]):
            grad_sum = sum(p.grad for p in params if p.grad is not None)
            for p in params:
                p.grad = grad_sum / n


# ---------------------------------------------------------------------------
# Model Parallelism
# ---------------------------------------------------------------------------

class ModelParallelModel(nn.Module):
    """Model (Tensor) Parallelism: single model split across multiple GPUs.

    How it works:
    - Different layers (or parts of layers) live on different GPUs.
    - A single batch flows through sequentially, being transferred between GPUs at each boundary.
    - Useful when a single model is too large for one GPU's memory.
    - Naive version has low GPU utilisation (one GPU active at a time) — pipeline parallelism fixes this.

    This illustrates layer-wise splitting (a simple 2-GPU case).
    """

    def __init__(self, layer_groups: List[nn.Sequential], device_ids: List[int]):
        super().__init__()
        assert len(layer_groups) == len(device_ids)
        self.stages = nn.ModuleList([
            group.to(torch.device(f"cuda:{did}"))
            for group, did in zip(layer_groups, device_ids)
        ])
        self.device_ids = device_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage, did in zip(self.stages, self.device_ids):
            x = stage(x.to(torch.device(f"cuda:{did}")))                   # transfer at each stage
        return x


# ---------------------------------------------------------------------------
# Pipeline Parallelism
# ---------------------------------------------------------------------------

class PipelineParallelModel(nn.Module):
    """Pipeline Parallelism: model split into stages across GPUs, micro-batches fill the pipeline.

    How it works:
    - Model is split into N sequential stages, one per GPU (same as model parallelism).
    - The batch is split into M micro-batches.
    - Each GPU starts processing the next micro-batch as soon as the previous one moves forward.
    - This keeps all GPUs active simultaneously, reducing the idle 'bubble' of naive model parallelism.
    - GPipe and PipeDream are the two main scheduling strategies.

    This is a simplified synchronous illustration without micro-batch overlap.
    For real pipeline parallelism use torch.distributed.pipeline.sync.Pipe or DeepSpeed.
    """

    def __init__(self, stages: List[nn.Module], device_ids: List[int], num_micro_batches: int = 4):
        super().__init__()
        assert len(stages) == len(device_ids)
        self.stages = nn.ModuleList([
            stage.to(torch.device(f"cuda:{did}"))
            for stage, did in zip(stages, device_ids)
        ])
        self.device_ids = device_ids
        self.num_micro_batches = num_micro_batches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        micro_batches = x.chunk(self.num_micro_batches, dim=0)
        outputs = []
        for mb in micro_batches:
            for stage, did in zip(self.stages, self.device_ids):
                mb = stage(mb.to(torch.device(f"cuda:{did}")))
            outputs.append(mb)
        return torch.cat(outputs, dim=0)
