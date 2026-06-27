# 007 — Cost-aware router + complex `program` task

**Goal:** проверить router по типам связей на более сложной осмысленной задаче и добавить стоимость каналов, чтобы модель не всегда использовала самые сильные/дорогие typed operators.

## Files

- Code: [`adfc/graph_adfc_worm_cost_router.py`](../adfc/graph_adfc_worm_cost_router.py)
- Program 240-step run: [`results/007_program_cost_router_L002_s240/`](../results/007_program_cost_router_L002_s240/)
- Cost sweep runs:
  - [`results/007_program_cost_router_L0.00_s180/`](../results/007_program_cost_router_L0.00_s180/)
  - [`results/007_program_cost_router_L0.01_s180/`](../results/007_program_cost_router_L0.01_s180/)
  - [`results/007_program_cost_router_L0.02_s180/`](../results/007_program_cost_router_L0.02_s180/)
  - [`results/007_program_cost_router_L0.05_s180/`](../results/007_program_cost_router_L0.05_s180/)
  - [`results/007_program_cost_router_L0.10_s180/`](../results/007_program_cost_router_L0.10_s180/)
  - [`results/007_program_cost_router_L0.20_s180/`](../results/007_program_cost_router_L0.20_s180/)
- Mixed run: [`results/007_mixed_cost_router_L002_s240/`](../results/007_mixed_cost_router_L002_s240/)

## New task: `program`

`program` is intentionally more compositional than earlier toy tasks.

Sequence contains:

```text
mode bit
A/B temporal events
4 key-value records
2 candidate query keys: q0 and q1
```

Rule:

```text
cond = A_before_B XOR mode
selected_key = q0 if cond else q1
label = value[selected_key]
```

So the model needs:

```text
order relation
conditional selection
key-value recall
```

## Cost-aware router

Channels:

```text
graph = sparse graph communication
order = PairwiseOrderBank
key   = KeyReadBank
```

Loss:

```text
loss = CE + entropy_penalty * H(router) + cost_lambda * expected_channel_cost
```

Default channel costs:

```text
graph = 0.05
key   = 0.35
order = 0.55
```

## Program-only run, 240 steps, lambda=0.02

Command output stored in:

```text
results/007_program_cost_router_L002_s240/
```

| Model | Best acc | Final acc | Final weights graph/order/key | Final expected cost |
|---|---:|---:|---:|---:|
| `graph` | 70.31% | 70.31% | 0.00 / 0.00 / 0.00 | 0.000 |
| `always_typed` | **83.30%** | 80.76% | 0.28 / 0.34 / 0.38 | 0.000 reported, fixed global mix |
| `cost_router` | 83.20% | **83.20%** | 0.00 / 0.00 / 1.00 | 0.350 |
| `cost_router_hard` | 81.35% | 81.35% | 0.01 / 0.02 / 0.97 | 0.350 |

### Interpretation

`program` is much harder than previous `route/order/kv` tasks. The graph-only model reaches only 70.31%, while typed models reach around 83%.

Unexpected result: with lambda=0.02, `cost_router` collapsed almost entirely to the key channel by the end of the 240-step run. It nearly matched `always_typed`, but did not use the order channel as much as expected.

## Program cost sweep, 180 steps

| cost_lambda | always_typed best | cost_router best | router cost | final weights graph/order/key | Winner |
|---:|---:|---:|---:|---:|---|
| 0.00 | **81.64%** | 81.54% | 0.407 | 0.00 / 0.30 / 0.70 | always_typed by 0.10% |
| 0.01 | 81.64% | **82.13%** | 0.406 | 0.00 / 0.29 / 0.71 | cost_router |
| 0.02 | 81.64% | **82.13%** | 0.401 | 0.00 / 0.26 / 0.74 | cost_router |
| 0.05 | 81.64% | **81.74%** | 0.385 | 0.00 / 0.19 / 0.81 | cost_router by 0.10% |
| 0.10 | **81.64%** | 80.08% | 0.350 | 0.00 / 0.00 / 1.00 | always_typed |
| 0.20 | **81.64%** | 80.47% | **0.132** | 0.71 / 0.00 / 0.29 | always_typed |

### Sweep interpretation

Low cost pressure (`lambda=0.01–0.02`) gives the best accuracy: 82.13%, slightly above always-typed 81.64%.

High cost pressure (`lambda=0.20`) forces the router to use cheap graph + key and completely drop order. Accuracy falls only to 80.47%, while expected cost drops strongly from about 0.40 to 0.132.

This is the first clear accuracy/cost tradeoff:

```text
lambda 0.02: 82.13% acc, cost 0.401
lambda 0.20: 80.47% acc, cost 0.132
```

## Mixed run: route/order/kv/program together

Run:

```text
results/007_mixed_cost_router_L002_s240/
```

| Model | Best mixed acc | Final mixed acc | Final route | Final order | Final kv | Final program | Final weights graph/order/key | Final expected cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `graph` | 69.92% | 69.92% | 89.5% | 48.4% | 68.8% | 73.0% | 0.00 / 0.00 / 0.00 | 0.000 |
| `always_typed` | **96.00%** | 95.02% | 100.0% | 100.0% | 100.0% | 80.1% | 0.26 / 0.37 / 0.37 | 0.000 reported, fixed global mix |
| `cost_router` | 95.41% | **95.41%** | 100.0% | 100.0% | 100.0% | 81.6% | 0.02 / 0.53 / 0.44 | 0.454 |
| `cost_router_hard` | 94.82% | 94.82% | 99.6% | 100.0% | 100.0% | 79.7% | 0.08 / 0.38 / 0.54 | 0.433 |

### Mixed interpretation

`always_typed` still has the best peak mixed accuracy, but `cost_router` has the best final `program` accuracy in the mixed run:

```text
always_typed final program = 80.1%
cost_router final program  = 81.6%
```

The router learned a different policy in mixed training than in program-only training:

```text
program-only lambda=0.02 final: graph 0.00 / order 0.00 / key 1.00
mixed lambda=0.02 final:        graph 0.02 / order 0.53 / key 0.44
```

So mixed training encourages a more balanced use of order and key operators.

## Main conclusion

The experiment confirms three things:

1. `program` is a more meaningful hard task than isolated `route/order/kv`.
2. A router over edge/operator types works, but it can collapse to whichever typed operator is easiest unless the task mix or cost forces specialization.
3. Cost-aware routing creates a real accuracy/compute tradeoff, but the current cost model is still too crude.

Current best settings:

```text
Best program-only accuracy: cost_router lambda=0.01 or 0.02, 82.13%
Best cheap program setting: lambda=0.20, 80.47% at cost 0.132
Best mixed peak accuracy: always_typed, 96.00%
Best mixed final program accuracy: cost_router lambda=0.02, 81.6%
```

## What failed / Что не получилось

We wanted:

```text
route   -> graph
order   -> order
kv      -> key
program -> order + key + maybe graph
```

But current router often chooses:

```text
program-only: mostly key
mixed: mostly order + key
```

Graph is still underused because order/key banks are too strong and can solve routing-like parts too.

## Next experiment

Need a budgeted/moe-style router where each channel has both a cost and a capacity constraint:

```text
1. top-k routing over channels
2. per-channel usage target or load-balancing
3. optional channel dropout during training
4. task-conditional diagnostics of channel usage
```

Hypothesis:

```text
channel dropout + cost + top-k should prevent collapse into key-only or typed-only solutions.
```
