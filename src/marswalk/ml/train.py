"""Rock-detector training entry point. UNTESTED ON REAL DATA (needs torch + GPU).

Targets the RTX 4050 (6 GB) laptop: small pretrained encoder, 256 px tiles,
batch 2-4, gradient accumulation, channels_last, logs peak VRAM and epoch time
(MARSWALK_CONTEXT.md §4). This file holds the config and a guarded entry point;
the actual model/loop is intentionally minimal and must be run and reported by
an owner. Evaluate with a spatial holdout (ml.splits) and
`precision_recall_by_size` (ml.metrics_det).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    annotations_json: str
    tile_size: int = 256
    batch_size: int = Field(default=2, ge=1, le=8)  # 6 GB -> 2-4
    grad_accum: int = Field(default=2, ge=1)
    encoder: str = "resnet18"  # small pretrained encoder
    epochs: int = 20
    lr: float = 1e-3
    amp: bool = True  # little speedup without tensor cores, saves memory
    channels_last: bool = True
    block_size_m: float = 50.0  # spatial holdout block size
    test_fraction: float = 0.3
    seed: int = 0
    out_dir: str = "runs/rock_detector"


def train(config: TrainConfig):  # pragma: no cover - needs torch + GPU, UNTESTED
    """Train the rock detector. Raises an actionable error without torch.

    An owner fills in the model (U-Net / Mask R-CNN with a small pretrained
    encoder) and the loop, then pastes back peak VRAM
    (`torch.cuda.max_memory_allocated()`), epoch time, and F1 by size bin on the
    spatial holdout.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ml.train.train requires torch + a CUDA GPU, not available in the "
            "sandbox. Run on the RTX 4050 laptop; this path is UNTESTED ON REAL "
            "DATA. See docs/ml.md for the recipe and what to report back."
        ) from exc

    raise NotImplementedError(
        "Model/loop to be implemented and run by the owner (P2). Config is "
        f"{config.model_dump()}. Log peak VRAM and epoch time; evaluate with a "
        "spatial holdout + precision_recall_by_size."
    )
