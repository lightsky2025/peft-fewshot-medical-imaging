# Team Interface Contract — How to Use This

## The idea

Instead of A → B → C working in sequence (a hidden pipeline that stalls
everyone until the person before them finishes), all three of you code
against `interface_contract.py` starting **today**, in parallel.

- `ExperimentConfig` — the input every method receives
- `RunResult` — the output every method must return
- `train_method(config)` — the one function signature Person A and
  Person B both implement (different `method_name` branches)
- `mock_train_method(config)` — a fake trainer returning realistic
  random results instantly, so Person C never has to wait

## Day 1 checklist

1. Everyone clones/copies `interface_contract.py` into their own branch.
2. Run it (`python interface_contract.py`) — confirms the contract
   itself works before anyone builds on it.
3. Agree as a team on any config fields you want to add or rename.
   **Do this once, together, before writing real code** — changing the
   contract later means re-touching everyone's work.

## Person A (Data & Baselines)

- Build the real `k_shot` / `n_way` sampler using `ExperimentConfig`'s
  dataset fields (`dataset_name`, `n_way`, `k_shot`, `seed`, `data_root`).
- Implement `train_method()` branches for `linear_probe`, `bitfit`,
  `layernorm` — must return a populated `RunResult`.
- Test each branch by swapping it in for `mock_train_method` one at a
  time and confirming the shape still matches.

## Person B (PEFT Methods)

- Implement `train_method()` branches for `lora`, `qlora`, using
  `config.method_kwargs` for method-specific settings (`r`, `alpha`,
  `quant_bits`, etc.).
- Use HuggingFace `peft` + `bitsandbytes` — don't hand-roll LoRA math.
- Same testing approach: swap into the real pipeline once ready.

## Person C (Profiling & Evaluation)

- Start immediately using `mock_train_method()` — you do not need to
  wait for A or B.
- Build:
  - `profile_run(result: RunResult) -> dict` — pulls
    `trainable_params`, `peak_gpu_memory_mb`, `train_time_seconds`,
    `flops` into a comparison table.
  - `evaluate(result: RunResult) -> dict` — computes accuracy, F1,
    AUROC, ECE from `y_true`, `y_pred`, `y_prob`.
  - Analysis notebook: accuracy-vs-compute scatter, Pareto frontier
    across all methods.
- When A/B's real methods are ready, change one line (swap
  `mock_train_method` for `train_method`) — nothing else changes.

## Integration point (end of week 2, suggested)

Once A and B have working `train_method()` branches for their methods,
merge into one shared `methods.py` with the dispatch pattern shown
in the docstring of `train_method()`. Run the full grid
(all methods × all datasets × 3-5 seeds) and hand results to Person C.

## Suggested grid

```
for dataset in [dataset_1, dataset_2]:
    for method in [linear_probe, bitfit, layernorm, lora, qlora]:
        for k_shot in [5, 10]:
            for seed in [0, 1, 2]:
                run train_method(config) -> RunResult
                save result to results/ as json or pickle
```

Person C's notebook then loads every saved `RunResult` and produces
the final comparison plots.
