# 009 — Metabolic structural plasticity

**Goal:** проверить принцип “связь/канал живёт, если польза больше цены”.

This is the first compact metabolic version. It is intentionally minimal and built on top of the already working CostAwareRouter from experiment 007.

## Files

- Code: [`adfc/graph_adfc_worm_metabolic.py`](../adfc/graph_adfc_worm_metabolic.py)
- One-command runner: [`run_009_metabolic_check.sh`](../run_009_metabolic_check.sh)
- Smoke result: [`results/009_smoke/`](../results/009_smoke/)
- Full result target: `results/009_metabolic_mixed_s240/`

## Mechanisms

Added to the previous router:

```text
channel survival gates
channel maintenance cost
sparse graph edge proxy cost
activity proxy cost
homeostasis proxy
metabolic diagnostics
```

The current version is compact: it does not yet implement full hard birth/death of individual edges. It tests whether metabolic pressure gives a useful signal.

## Loss

```text
loss = CE
     + cost_lambda    * expected_channel_cost
     + target_lambda  * target_usage_loss
     + channel_lambda * channel_alive_cost
     + edge_lambda    * edge_proxy_cost
     + activity_lambda * activity_proxy
     + homeo_lambda   * homeostasis_loss
```

## Smoke test

Smoke ran on Tesla P40 and produced:

```text
results/009_smoke/summary_rows.csv
```

Example smoke values after 3 steps:

```text
val_acc = 49.22%
best_val_acc = 53.12%
expected_cost = 0.3428
metabolic_loss = 0.0633
survive_graph = 0.7848
survive_order = 0.7858
survive_key = 0.7849
edge_proxy_cost = 0.4996
activity_proxy = 0.5759
```

This is only a runtime check, not a quality benchmark.

## One-command full check

From repository root:

```bash
./run_009_metabolic_check.sh
```

It runs:

```text
mixed task = route + order + kv + program
models = always_typed, cost_router, metabolic, metabolic_hard
steps = 240
```

## What to inspect

Main metrics:

```text
best_val_acc
val_acc
acc_route / acc_order / acc_kv / acc_program
expected_cost
metabolic_loss
survive_graph / survive_order / survive_key
edge_proxy_cost
activity_proxy
w_graph / w_order / w_key
```

Desired behavior:

```text
accuracy close to cost_router / always_typed
lower expected cost or lower survival/edge/activity cost
less collapse into a single channel
useful specialization remains
```

## Interpretation target

If metabolic accuracy is close but costs/survival gates drop, the principle is useful.

If accuracy collapses, metabolic penalties are too strong.

If costs do not drop, the penalties are too weak or the proxies are not connected strongly enough to computation.

## Next step after this

If compact 009 works, then 010 should implement real structural plasticity:

```text
hard edge gates
utility EMA = abs(message * grad_receiver)
prune edges with utility < cost
birth new candidate edges from residual/correlation
```
