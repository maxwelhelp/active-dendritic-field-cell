# 008 — Budgeted top-k router

**Goal:** исправить collapse из experiment 007, где router часто уходил в самый удобный typed channel (`key` или `order+key`) и почти не использовал cheap graph.

## Files

- Code: [`adfc/graph_adfc_worm_budgeted_router.py`](../adfc/graph_adfc_worm_budgeted_router.py)
- One-command runner: [`run_008_budgeted_check.sh`](../run_008_budgeted_check.sh)
- Smoke result: [`results/008_smoke/`](../results/008_smoke/)
- Full result target: `results/008_budgeted_topk_mixed_s240/`

## What changed vs 007

Experiment 007 had:

```text
cost_router = soft router + expected channel cost
```

Experiment 008 adds anti-collapse mechanisms:

```text
top-k channel routing
channel dropout during training
target usage loss by task type
global usage/balance loss
```

Channels:

```text
graph = sparse graph communication
order = PairwiseOrderBank
key   = KeyReadBank
```

Expected specialization target:

| Task | graph | order | key |
|---|---:|---:|---:|
| `route` | 0.75 | 0.10 | 0.15 |
| `order` | 0.05 | 0.90 | 0.05 |
| `kv` | 0.05 | 0.15 | 0.80 |
| `program` | 0.05 | 0.45 | 0.50 |

## Loss

```text
loss = CE
     + cost_lambda    * expected_channel_cost
     + entropy_penalty * router_entropy
     + target_lambda  * per-sample target usage MSE
     + balance_lambda * batch/global usage MSE
```

Default check settings:

```text
top_k = 2
channel_dropout = 0.12
cost_lambda = 0.02
target_lambda = 0.08
balance_lambda = 0.04
entropy_penalty = 0.005
```

## Smoke test

Smoke command used only 3 steps and only `budget_topk`, to check runtime correctness.

Result:

```text
compile_ok
smoke runs on Tesla P40
budget_topk writes weights / target_loss / per-task metrics
```

Early smoke example:

```text
step 1: mixed val=48.44%, cost=0.334, target_loss=0.169
step 3: mixed val=46.09%, cost=0.391, target_loss=0.115
```

This is not a quality result; it only confirms the code runs.

## Full check command

Use one command from repository root:

```bash
./run_008_budgeted_check.sh
```

It runs:

```text
mixed task = route + order + kv + program
models = graph, always_typed, cost_router, budget_topk, budget_topk_hard
steps = 240
```

## What to inspect after full run

Main fields:

```text
best_val_acc
val_acc
acc_route
acc_order
acc_kv
acc_program
expected_cost
target_loss
w_graph / w_order / w_key
w_graph_route / w_order_order / w_key_kv / w_order_program / w_key_program
```

Desired result:

```text
budget_topk accuracy close to always_typed / cost_router
expected_cost lower or channel usage more specialized
route uses more graph than previous router
order uses order
kv uses key
program uses order + key
```

## Expected failure mode

If accuracy drops too much, reduce target pressure:

```text
target_lambda 0.08 -> 0.03
balance_lambda 0.04 -> 0.01
channel_dropout 0.12 -> 0.05
```

If graph is still ignored, increase anti-collapse pressure:

```text
target_lambda 0.08 -> 0.15
channel_dropout 0.12 -> 0.20
```
