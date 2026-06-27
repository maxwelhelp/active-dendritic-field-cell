# 002 — ADFC6 directional order kernel

**Goal:** исправить слабое место ADFC3: порядок событий `A before B`.

## Files

- Code: [`adfc/adfc_v6_orderkernel.py`](../adfc/adfc_v6_orderkernel.py)
- Results: [`results/v6_order/`](../results/v6_order/)
- Summary: [`RESULTS.md`](../RESULTS.md)

## What was tested

ADFC6 = ADFC3 + directional order kernel:

```text
order = A_before_B - B_before_A
```

## Result

| Task | mean_pool | ADFC3 | ADFC6 |
|---|---:|---:|---:|
| `order_compare` | 51.37% | 86.96% | **99.85%** |

## Conclusion

Temporal order is not solved well by plain keyed memory. A directional relation operator solves it almost perfectly on this synthetic task.
