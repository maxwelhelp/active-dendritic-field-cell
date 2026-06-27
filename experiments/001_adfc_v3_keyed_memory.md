# 001 — ADFC3 keyed memory

**Goal:** проверить, чинит ли адресное чтение провал ADFC2 на `kv_recall4`.

## Files

- Code: [`adfc/adfc_v3_keyed.py`](../adfc/adfc_v3_keyed.py)
- Results: [`results/v3_targeted/`](../results/v3_targeted/)
- Summary: [`RESULTS.md`](../RESULTS.md)

## What was tested

ADFC3 = active dendritic state + keyed read:

```text
tokens -> dendritic state -> keyed memory read -> classifier
```

## Result

| Task | baseline | ADFC3 |
|---|---:|---:|
| `mode_select` | 75.78% | **100.00%** |
| `order_compare` | 51.66% | 54.20% |
| `kv_recall4` | 71.48% | **100.00%** |

## Conclusion

Keyed memory solves associative recall and mode selection, but does not solve temporal order.
