"""
==================================================================
 SHARED INTERFACE CONTRACT
 Compute-Aware Comparison of PEFT Methods for Few-Shot Medical
 Image Classification
==================================================================

WHY THIS FILE EXISTS
---------------------
This defines the ONE function signature and data format that all
three people code against. If everyone respects these shapes, you
can build your three pieces in parallel starting today, using the
MockMethod at the bottom instead of waiting for real code.

Owners:
  Person A -> implements train_method() for: linear_probe, bitfit, layernorm
  Person B -> implements train_method() for: lora, qlora
  Person C -> implements everything in evaluate() and profile_run(),
              and consumes RunResult objects

DO NOT change these shapes without agreeing as a team first —
that's the whole point of a contract.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time
import random


# ------------------------------------------------------------------
# 1. CONFIG SCHEMA  (Person A defines dataset side, Person B method side,
#    but everyone imports this same class)
# ------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # --- dataset / few-shot setup (Person A owns these fields) ---
    dataset_name: str            # e.g. "isic", "chestxray", "medmnist_derma"
    n_way: int                   # number of classes sampled per episode
    k_shot: int                  # number of support examples per class (5, 10, ...)
    seed: int                    # random seed for this run (use 3-5 seeds per config)
    data_root: str = "./data"

    # --- backbone (shared, agree on this together) ---
    backbone_name: str = "vit_base_patch16_224"   # or "biomedclip", "resnet50", etc.
    pretrained: bool = True

    # --- PEFT method (Person B owns lora/qlora fields, Person A owns the rest) ---
    method_name: str = "linear_probe"   # one of: linear_probe, bitfit, layernorm,
                                         #         lora, qlora, mock
    method_kwargs: Dict[str, Any] = field(default_factory=dict)
    # examples of what goes in method_kwargs:
    #   lora    -> {"r": 8, "alpha": 16, "target_modules": ["q_proj","v_proj"]}
    #   qlora   -> {"r": 8, "alpha": 16, "quant_bits": 4}
    #   bitfit  -> {}   (nothing to configure)

    # --- training (shared defaults, override per method if needed) ---
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 16
    device: str = "cuda"


# ------------------------------------------------------------------
# 2. OUTPUT SHAPE — every train_method() call must return exactly this
# ------------------------------------------------------------------

@dataclass
class RunResult:
    config: ExperimentConfig

    # what Person C's profiler needs (filled during/after training)
    trainable_params: int = 0
    total_params: int = 0
    peak_gpu_memory_mb: float = 0.0
    train_time_seconds: float = 0.0
    flops: Optional[float] = None

    # what Person C's evaluator needs (predictions on the held-out query set)
    y_true: List[int] = field(default_factory=list)
    y_pred: List[int] = field(default_factory=list)
    y_prob: List[List[float]] = field(default_factory=list)  # softmax probs, for AUROC/ECE

    # free-form extras (loss curves, notes, etc.) — anything goes here,
    # not part of the strict contract
    extra: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# 3. THE ONE FUNCTION SIGNATURE EVERYONE IMPLEMENTS AGAINST
# ------------------------------------------------------------------

def train_method(config: ExperimentConfig) -> RunResult:
    """
    THE contract. Person A implements this for linear_probe/bitfit/layernorm.
    Person B implements this for lora/qlora.

    Dispatch pattern (put this in your real implementation file):

        def train_method(config):
            if config.method_name == "linear_probe":
                return _train_linear_probe(config)
            elif config.method_name == "bitfit":
                return _train_bitfit(config)
            elif config.method_name == "lora":
                return _train_lora(config)
            ...
            elif config.method_name == "mock":
                return _mock_train(config)   # see below — use this to test early
            else:
                raise ValueError(f"unknown method {config.method_name}")

    Whatever you do inside, you MUST return a fully-populated RunResult.
    """
    raise NotImplementedError("Implement per-method training here")


# ------------------------------------------------------------------
# 4. MOCK METHOD — Person C uses this from Day 1, no need to wait
#    for A or B to finish real training code.
# ------------------------------------------------------------------

def mock_train_method(config: ExperimentConfig) -> RunResult:
    """
    Fake trainer that returns realistic-shaped random results instantly.
    Person C: build your entire profiling + evaluation harness against
    this function. When A/B finish real methods, you just swap this
    call for the real train_method() — nothing else in your code changes.
    """
    random.seed(config.seed)
    n_samples = config.n_way * 20  # pretend 20 query examples per class

    y_true = [random.randint(0, config.n_way - 1) for _ in range(n_samples)]
    y_pred = [
        yt if random.random() < 0.7 else random.randint(0, config.n_way - 1)
        for yt in y_true
    ]
    y_prob = []
    for yp in y_pred:
        probs = [random.random() * 0.3 for _ in range(config.n_way)]
        probs[yp] += 0.6
        s = sum(probs)
        y_prob.append([p / s for p in probs])

    fake_param_counts = {
        "linear_probe": (2_000, 86_000_000),
        "bitfit": (40_000, 86_000_000),
        "layernorm": (30_000, 86_000_000),
        "lora": (300_000, 86_000_000),
        "qlora": (300_000, 86_000_000),
        "mock": (1_000, 86_000_000),
    }
    trainable, total = fake_param_counts.get(config.method_name, (1_000, 86_000_000))

    return RunResult(
        config=config,
        trainable_params=trainable,
        total_params=total,
        peak_gpu_memory_mb=random.uniform(2000, 12000),
        train_time_seconds=random.uniform(30, 600),
        flops=random.uniform(1e9, 5e10),
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        extra={"note": "mock run, not real training"},
    )


# ------------------------------------------------------------------
# 5. QUICK SELF-TEST — run this file directly to confirm the contract
#    works end-to-end before anyone writes real code.
# ------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ExperimentConfig(
        dataset_name="isic",
        n_way=5,
        k_shot=5,
        seed=42,
        method_name="lora",
        method_kwargs={"r": 8, "alpha": 16},
    )
    result = mock_train_method(cfg)
    print("Contract self-test passed.")
    print(f"  method: {result.config.method_name}")
    print(f"  trainable params: {result.trainable_params:,}")
    print(f"  peak GPU mem (MB): {result.peak_gpu_memory_mb:.1f}")
    print(f"  train time (s): {result.train_time_seconds:.1f}")
    print(f"  num predictions: {len(result.y_pred)}")
