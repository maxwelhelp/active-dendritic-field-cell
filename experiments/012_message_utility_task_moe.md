# 012 — Message-utility Task-MoE

Goal: improve 011 by adding best checkpoint saving and stronger graph utility.

## Files

- Code: `adfc/graph_adfc_worm_task_moe_msgutil.py`
- Runner: `run_012_msgutil_check.sh`
- Smoke: `results/012_smoke/`
- Full output: `results/012_task_moe_msgutil_s160/`

## What changed vs 011

011 already showed clear task/channel specialization and reached 96.78% peak, but graph utility stayed zero.

012 changes:

- default steps = 160, because 011 peaked at step 160 and degraded by step 200
- saves `best_<model>.pt` checkpoint when validation improves
- graph utility uses message-gradient signal instead of only task gate gradient
- structural update receives actual mean task probabilities from the current batch

## Utility approximation

During graph forward, the model stores graph messages and attention-like edge weights. After backward, utility is estimated from message influence:

```text
receiver_utility = abs(message * grad(message))
edge_utility ~= edge_weight * receiver_utility
```

This is still approximate, but closer to true edge usefulness than the older gate-gradient proxy.

## Command

```bash
./run_012_msgutil_check.sh
```

It compares only:

- `always_typed`
- `task_moe_msgutil`

## Main metrics

Inspect:

- `best_val_acc`
- `acc_program`
- `taskp_*_*`
- `w_graph_*`, `w_order_*`, `w_key_*`
- `graph_gate_route/order/kv/program`
- `edge_util_route/order/kv/program`
- saved `best_*.pt`

## Smoke

Smoke passed on Tesla P40. It confirms the retained message gradients do not break runtime.

## Full result

Run directory:

```text
results/012_task_moe_msgutil_s160/
```

Compared:

```text
always_typed
task_moe_msgutil
```

### Accuracy

| Model | Best mixed | Final mixed | route | order | kv | program |
|---|---:|---:|---:|---:|---:|---:|
| `always_typed` | 95.12% | 95.12% | 100.0% | 100.0% | 100.0% | 80.47% |
| `task_moe_msgutil` | **95.80%** | **95.80%** | 100.0% | 100.0% | 100.0% | **83.20%** |

Both models saved best checkpoints:

```text
best_always_typed.pt
best_task_moe_msgutil.pt
```

### Specialization

Final task probabilities:

| Input task | p(route) | p(order) | p(kv) | p(program) |
|---|---:|---:|---:|---:|
| `route` | 0.072 | 0.436 | **0.474** | 0.018 |
| `order` | 0.002 | **0.994** | 0.003 | 0.002 |
| `kv` | 0.008 | 0.236 | **0.703** | 0.053 |
| `program` | 0.005 | 0.020 | 0.017 | **0.958** |

Channel weights:

| Input task | graph | order | key |
|---|---:|---:|---:|
| `route` | 0.101 | 0.479 | 0.421 |
| `order` | 0.051 | **0.896** | 0.053 |
| `kv` | 0.056 | 0.343 | **0.602** |
| `program` | 0.054 | 0.452 | **0.495** |

### Graph utility result

Message-gradient utility is no longer exactly zero, but it is still extremely tiny:

```text
edge_util_route   = 9.9e-14
edge_util_order   = 1.86e-12
edge_util_kv      = 1.51e-12
edge_util_program = 1.47e-12
```

Graph gates differ slightly by task:

```text
graph_gate_route   = 0.0867
graph_gate_order   = 0.0753
graph_gate_kv      = 0.0786
graph_gate_program = 0.0765
```

### Interpretation

012 improves over `always_typed` and preserves strong specialization. It also saves best checkpoints. However, graph edge utility is still too small to prove real physical graph plasticity.

What is proven:

```text
task/channel specialization works
program improves over fixed always_typed
best checkpoint saving works
```

What is not proven yet:

```text
true task-specific graph substructure
strong useful-edge growth
message utility signal is still too weak
```

Next step should make graph structurally necessary, otherwise the order/key expert channels solve most of the task and graph edge utility stays tiny.
