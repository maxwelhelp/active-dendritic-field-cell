# Experiment index / Индекс экспериментов

Короткий журнал. Один файл = один эксперимент. Каждый файл ссылается на код и папку результатов.

| ID | Experiment | Main idea | Result |
|---|---|---|---|
| 001 | [ADFC3 keyed memory](experiments/001_adfc_v3_keyed_memory.md) | dendritic state + keyed read | solves `mode_select` and `kv_recall4` |
| 002 | [ADFC6 order kernel](experiments/002_adfc_v6_order_kernel.md) | directional order operator | `order_compare` 99.85% |
| 003 | [Graph connections](experiments/003_graph_connections.md) | compare graph wiring variants | graph helps route/kv, not order |
| 004 | [Graph + first typed ops](experiments/004_graph_plus_typed_ops.md) | typed key/order channels | key helps, order weak |
| 005 | [Graph TypedBank](experiments/005_graph_typed_bank.md) | pairwise relation bank + key read | route/order/kv ≈ 100% |
| 006 | [Edge-type router](experiments/006_edge_type_router.md) | learned per-sample routing over graph/order/key | router 99.61% mixed, selective but ignores graph |
| 007 | [Cost-aware router + program task](experiments/007_cost_aware_router_program.md) | compositional program task + channel cost | cost-router trades accuracy for lower expected cost |
| 008 | [Budgeted top-k router](experiments/008_budgeted_topk_router.md) | top-k routing + channel dropout + target usage | code + smoke ready; full one-command check pending |
| 009 | [Metabolic structural plasticity](experiments/009_metabolic_structural_plasticity.md) | survival gates + metabolic cost proxies | metabolic nearly matches always_typed with lower expected cost; hard metabolic hurts order |
| 010 | [Utility structural plasticity](experiments/010_utility_structural_plasticity.md) | edge utility EMA + reward/decay + birth/prune | code + one-command runner ready; full check pending |
| 011 | [Task-MoE specialization](experiments/011_task_moe_specialization.md) | task probabilities + task-conditioned graph subgates | code + smoke ready; one-command full check pending |

## Current conclusion

```text
cell dynamics matter
connections matter
topology matters
edge/operator types matter most for relation tasks
routing over edge types works, but needs cost/budget to prefer cheap graph channels
cost-aware routing creates accuracy/cost tradeoff on compositional program task
budgeted top-k routing adds anti-collapse pressure and target channel specialization
metabolic routing adds survival gates and resource pressure as a first step toward structural plasticity
```

## Current best file

- [`adfc/graph_adfc_worm_typed_bank.py`](adfc/graph_adfc_worm_typed_bank.py)

## Current best result

- [`results/graph_typed_bank_full/`](results/graph_typed_bank_full/)
