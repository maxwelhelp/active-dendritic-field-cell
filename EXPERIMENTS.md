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

## Current conclusion

```text
cell dynamics matter
connections matter
topology matters
edge/operator types matter most for relation tasks
routing over edge types works, but needs cost/budget to prefer cheap graph channels
```

## Current best file

- [`adfc/graph_adfc_worm_typed_bank.py`](adfc/graph_adfc_worm_typed_bank.py)

## Current best result

- [`results/graph_typed_bank_full/`](results/graph_typed_bank_full/)
