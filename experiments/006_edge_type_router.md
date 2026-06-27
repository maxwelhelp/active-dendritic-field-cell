# 006 — Edge-type router over communication channels

**Goal:** проверить router по типам связей, чтобы модель не всегда складывала все typed channels, а сама выбирала между:

```text
graph  = sparse chemical graph communication
order  = PairwiseOrderBank\_key    = KeyReadBank
```

## Files

- Code: [`adfc/graph_adfc_worm_router.py`](../adfc/graph_adfc_worm_router.py)
- Results: [`results/graph_router_full/`](../results/graph_router_full/)
- Smoke: [`results/graph_router_smoke/`](../results/graph_router_smoke/)

## What was tested

Mixed training: every batch contains all three tasks:

```text
route
order
kv
```

Compared models:

```text
graph         # sparse graph only
always_typed  # fixed global mixture of graph/order/key
router        # learned soft per-sample router
router_hard   # straight-through hard-ish router
```

## Result

| Model | Mixed acc | route | order | kv | Router entropy |
|---|---:|---:|---:|---:|---:|
| `graph` | 65.62% best / 63.28% final | 71.18% | 51.47% | 67.15% | 0.00 |
| `always_typed` | **99.71%** | 99.71% | 100.00% | 99.42% | 1.00 |
| `router` | 99.61% | **100.00%** | **100.00%** | 98.84% | 0.40 |
| `router_hard` | 89.16% | 99.41% | 75.29% | 92.73% | 0.24 |

## Learned router weights by task

Final soft-router weights:

| Task | graph | order | key |
|---|---:|---:|---:|
| `route` | 0.002 | 0.436 | 0.562 |
| `order` | 0.002 | **0.987** | 0.012 |
| `kv` | 0.004 | 0.301 | **0.695** |

## Conclusion

The router works: it learns non-uniform per-sample channel selection and almost matches `always_typed`.

Important unexpected result:

```text
router did NOT choose graph for route.
```

Reason: `PairwiseOrderBank` and `KeyReadBank` are strong enough to solve `route` too, so the optimizer treats graph as unnecessary.

Current interpretation:

```text
edge-type routing works,
but to force cheap graph use for route we need channel costs / budget regularization.
```

## Next experiment

Add cost-aware routing:

```text
cost(graph) < cost(key) < cost(order)
loss += lambda * expected_channel_cost
```

Expected behavior:

```text
route -> graph if graph is accurate enough and cheaper
order -> order bank despite cost
kv    -> key bank despite cost
```
