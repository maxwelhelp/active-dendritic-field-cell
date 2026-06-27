# 005 — GraphADFC TypedBank relation operators

**Goal:** заменить слабый learned order detector на общий банк отношений по всем сенсорным каналам.

## Files

- Code: [`adfc/graph_adfc_worm_typed_bank.py`](../adfc/graph_adfc_worm_typed_bank.py)
- Results: [`results/graph_typed_bank_full/`](../results/graph_typed_bank_full/)
- Smoke: [`results/graph_typed_bank_smoke/`](../results/graph_typed_bank_smoke/)
- Summary: [`GRAPH_RESULTS.md`](../GRAPH_RESULTS.md)

## What was tested

```text
PairwiseOrderBank:
  for every sensory pair i,j:
    i_before_j - j_before_i

KeyReadBank:
  query-conditioned shifted-value read
```

## Result

| Task | none | learned_sparse | learned_sparse_typed |
|---|---:|---:|---:|
| `route` | 51.86% | 76.07% | **100.00%** |
| `order` | 51.86% | 52.54% | **99.90%** |
| `kv` | 51.07% | 66.60% | **99.90%** |

## Conclusion

This is the strongest current result. It supports the idea that connections need **types/operators**, not only scalar weights.
