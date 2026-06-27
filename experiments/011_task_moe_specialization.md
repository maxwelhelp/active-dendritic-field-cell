# 011 — Task-MoE specialization

Goal: make specialization explicit and faster to test.

Instead of running every old mode again, this experiment compares only:

- `always_typed`: known strong baseline
- `task_moe_utility`: new task-specialized model

## Files

- Code: `adfc/graph_adfc_worm_task_moe.py`
- Runner: `run_011_task_moe_check.sh`
- Smoke: `results/011_smoke/`
- Full output: `results/011_task_moe_mixed_s200/`

## What is new

`task_moe_utility` learns:

- task probabilities over `route/order/kv/program`
- channel mix from task probabilities
- task-conditioned graph subgates
- utility updates for task graph gates

This means we can inspect whether tasks use different structures, not only different final answers.

## What to inspect

Main columns:

- `task_p_route`, `task_p_order`, `task_p_kv`, `task_p_program`
- `w_graph`, `w_order`, `w_key`
- `taskp_*_route`, `taskp_*_order`, `taskp_*_kv`, `taskp_*_program`
- `graph_gate_route`, `graph_gate_order`, `graph_gate_kv`, `graph_gate_program`
- `edge_util_route`, `edge_util_order`, `edge_util_kv`, `edge_util_program`
- per-task accuracy

## Command

```bash
./run_011_task_moe_check.sh
```

## Desired behavior

- `order` should route toward order logic.
- `kv` should route toward key logic.
- `program` should use a mixed route.
- graph gates/utilities should diverge between task types.

## Smoke

The smoke run passed on Tesla P40 and wrote task probabilities/channel weights/per-task metrics.

## Full result

Run directory:

```text
results/011_task_moe_mixed_s200/
```

Compared only:

```text
always_typed
task_moe_utility
```

### Accuracy

| Model | Best mixed | Final mixed | route | order | kv | program |
|---|---:|---:|---:|---:|---:|---:|
| `always_typed` | 95.12% | 94.82% | 100.0% | 100.0% | 100.0% | 79.3% |
| `task_moe_utility` | **96.78%** | 94.53% | 100.0% | 99.6% | 100.0% | 78.5% final / **88.3% at best step** |

Peak result happened at step 160:

```text
task_moe_utility mixed = 96.78%
program at step 160 = 88.3%
```

### Final task specialization

At final step, the learned task probabilities were clearly different by task:

| Input task | p(route) | p(order) | p(kv) | p(program) |
|---|---:|---:|---:|---:|
| route | **0.460** | 0.270 | 0.266 | 0.004 |
| order | 0.002 | **0.996** | 0.001 | 0.001 |
| kv | 0.006 | 0.201 | **0.782** | 0.012 |
| program | 0.006 | 0.043 | 0.020 | **0.932** |

Channel weights derived from the task probabilities:

| Input task | graph | order | key |
|---|---:|---:|---:|
| route | **0.372** | 0.331 | 0.297 |
| order | 0.052 | **0.897** | 0.051 |
| kv | 0.054 | 0.304 | **0.642** |
| program | 0.054 | 0.461 | **0.485** |

### Interpretation

This is the clearest specialization result so far:

```text
order   -> almost pure order expert
kv      -> key expert + some order support
program -> program expert -> order+key mixture
route   -> partial route expert, still mixed with order/key
```

The new model beats the fixed `always_typed` baseline at peak accuracy and shows task-dependent computation paths.

What is still weak:

```text
final accuracy drops after peak
edge_util_* is still zero, so task-conditioned graph utility updates are not yet effective
route expert is not clean enough
```

Next step should preserve the best checkpoint or add early-stop, and fix graph utility logging/update so graph subgates truly specialize structurally, not only through task/channel routing.
