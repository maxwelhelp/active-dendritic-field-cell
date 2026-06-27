# 004 — GraphADFC + first typed ops

**Goal:** проверить, помогают ли typed operator channels поверх sparse graph.

## Files

- Code: [`adfc/graph_adfc_worm_plus.py`](../adfc/graph_adfc_worm_plus.py)
- Results: [`results/graph_plus_smoke/`](../results/graph_plus_smoke/)
- Summary: [`GRAPH_RESULTS.md`](../GRAPH_RESULTS.md)

## What was tested

```text
DirectionalOrderChannel
KeyedReadChannel
```

## Result

| Task | none | learned_sparse | learned_sparse_ops |
|---|---:|---:|---:|
| `order` | 50.52% | 50.65% | **52.34%** |
| `kv` | 50.00% | 52.86% | **69.66%** |

## Conclusion

Typed key-read helps immediately. The first learned order channel is too weak: its detector does not expose the temporal relation clearly.
