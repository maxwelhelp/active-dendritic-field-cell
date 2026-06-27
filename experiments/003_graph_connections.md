# 003 — GraphADFC connection benchmark

**Goal:** проверить гипотезу: часть “ума” находится в связях между клетками/узлами, а не только внутри клетки.

## Files

- Code: [`adfc/graph_adfc_worm.py`](../adfc/graph_adfc_worm.py)
- Results: [`results/graph_full_v1/`](../results/graph_full_v1/)
- Summary: [`GRAPH_RESULTS.md`](../GRAPH_RESULTS.md)

## Variants

```text
none
fixed_random
learned_dense
learned_sparse
chem_gap
```

## Result

| Task | none | best graph | Winner |
|---|---:|---:|---|
| `route` | 50.88% | **78.22%** | fixed_random |
| `order` | 51.76% | ~51% | no useful graph winner |
| `kv` | 53.03% | **67.97%** | learned_sparse |

## Conclusion

Graph connections matter for routing and partial recall. Plain scalar graph edges do not solve temporal order.
