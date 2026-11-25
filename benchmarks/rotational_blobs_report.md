# Rotational extrapolation: initial CPU ablation

Run from the repository root:

```bash
python -m benchmarks.benchmark_rotational_blobs --output rotational_blobs.csv
```

Tested implementation: 2468172012237aae80435299d95086a1501b00ec. CPU runner, recipe and extrapolation module were copied from GitHub and verified by Git blob hashes. PyTorch 2.10.0+cpu, scikit-learn 1.8.0, one PyTorch CPU thread. No GPU tests. 20,000 training points, 5,000 held-out points, 32 true and fitted clusters, dimensions 8 and 64, standard deviations 1 and 4, 5 paired algorithm seeds (fixed dataset per scenario), 4 epoch budgets. 240 fits total.

Mean paired relative reduction in held-out mean squared Euclidean distance (positive is better), median paired training runtime ratio:

| Epochs | Every 5 batches: gain % | Runtime ratio | Every batch: gain % | Runtime ratio |
|---|---:|---:|---:|---:|
| 0.25 | +0.087 | 1.278 | -1.342 | 2.040 |
| 0.50 | +0.023 | 1.276 | +0.044 | 2.002 |
| 1.00 | +0.043 | 1.264 | +0.102 | 2.015 |
| 2.00 | +0.099 | 1.273 | -0.061 | 1.827 |
| Overall | +0.063 | 1.274 | -0.314 | 1.969 |

No compelling improvement in this experiment. Gains mainly appear in the 8-dimensional overlapping regime (std=4). The final-pass averaging can suppress the direct influence of accelerated EMA centroids; the current mini-batch SSE acceptance may favor sampling noise. These are hypotheses, not established explanations. Next experiments should track accepted corrections, compare finalization on/off in a controlled factorial study, and use smoothed or windowed trajectories and more difficult distributions. Keep the feature opt-in.
