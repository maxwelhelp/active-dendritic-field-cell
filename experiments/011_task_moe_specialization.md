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
