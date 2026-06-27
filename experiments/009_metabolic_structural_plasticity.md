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

This only confirmed runtime correctness.

## Full result

Full run:

```bash
./run_009_metabolic_check.sh
```

Run directory:

```text
results/009_metabolic_mixed_s240/
```

Task mix:

```text
route + order + kv + program
```

Models:

```text
always_typed
cost_router
metabolic
metabolic_hard
```

### Accuracy

| Model | Best mixed | Final mixed | route | order | kv | program |
|---|---:|---:|---:|---:|---:|---:|
| `always_typed` | **95.80%** | **95.80%** | 100.00% | 100.00% | 100.00% | **83.20%** |
| `cost_router` | **95.80%** | 94.82% | 100.00% | 100.00% | 100.00% | 79.30% |
| `metabolic` | 95.51% | 95.51% | 100.00% | 100.00% | 100.00% | 82.03% |
| `metabolic_hard` | 88.18% | 87.70% | 99.61% | 71.48% | 100.00% | 79.69% |

### Cost / survival diagnostics

| Model | expected_cost | metabolic_loss | graph/order/key weights | survival graph/order/key | edge_proxy | activity_proxy |
|---|---:|---:|---:|---:|---:|---:|
| `cost_router` | 0.4066 | 0.0000 | 0.127 / 0.458 / 0.415 | - | - | - |
| `metabolic` | 0.3977 | 0.0552 | 0.133 / 0.431 / 0.436 | 0.714 / 0.725 / 0.737 | 0.3846 | 0.5336 |
| `metabolic_hard` | **0.3859** | 0.0531 | 0.191 / 0.184 / 0.625 | 0.689 / 0.690 / 0.690 | 0.3843 | 0.5060 |

## Interpretation

`always_typed` still wins peak accuracy. The compact `metabolic` model does **not** beat it.

But the metabolic model is useful because it keeps almost the same quality while reducing cost signals:

```text
cost_router final acc = 94.82%, expected_cost = 0.4066
metabolic final acc   = 95.51%, expected_cost = 0.3977
```

It also reduced channel survival gates from the initial ~0.79 range to:

```text
graph  = 0.714
order  = 0.725
key    = 0.737
```

So metabolic pressure is doing something real: channels are not free anymore, and survival probabilities move down while quality stays high.

The hard metabolic router is too aggressive:

```text
metabolic_hard order acc = 71.48%
```

It saves a bit more expected cost but damages order reasoning.

## Main conclusion

Compact metabolic pressure works as a **soft resource pressure**, but not yet as true structural plasticity.

What worked:

```text
metabolic survival gates decrease
expected channel cost decreases slightly
accuracy remains close to always_typed
program stays strong at 82.03%
```

What did not work yet:

```text
no explicit reward/utility for individual useful edges
no real hard edge birth/death
edge_proxy is only a proxy, not true connection utility
hard routing hurts order
```

## Next step

010 should implement real usefulness-based structural plasticity:

```text
edge_utility_ema = EMA(abs(message_ij * grad_receiver_i))
survival_score = utility - cost
if utility > cost: gate_ij gets bonus
if utility < cost: gate_ij decays/prunes
birth new candidate edge from residual/correlation
```

This is the missing positive side: 009 has mostly costs, 010 should add explicit reward for useful links.
