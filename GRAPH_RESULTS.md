# Graph-ADFC-Worm results / Результаты Graph-ADFC-Worm

This document records the first connection-centric experiments.

Этот документ фиксирует первые эксперименты, где проверяется гипотеза:

> может быть, главный интеллект сети находится не только в нейроне, а в структуре связей.

---

## Implemented files / Файлы

```text
adfc/graph_adfc_worm.py       # graph-only variants
adfc/graph_adfc_worm_plus.py  # graph + typed operator channels
```

Run directories:

```text
results/graph_full_v1/
results/graph_plus_smoke/
```

Hardware:

```text
GPU: NVIDIA Tesla P40
PyTorch: 2.6.0+cu124
```

---

## Experiment 1: graph-only connections

Command:

```bash
python -u adfc/graph_adfc_worm.py \
  --out results/graph_full_v1 \
  --steps 120 \
  --batch 192 \
  --eval-batch 256 \
  --tasks route,order,kv \
  --variants none,fixed_random,learned_dense,learned_sparse,chem_gap
```

Variants:

| Variant | Meaning |
|---|---|
| `none` | no communication from sensors to motor nodes |
| `fixed_random` | fixed sparse directed chemical graph |
| `learned_dense` | learned dense directed graph |
| `learned_sparse` | learned weights on fixed sparse candidate graph |
| `chem_gap` | learned chemical + gap/diffusion + global modulator |

### Results

| Task | none | fixed_random | learned_dense | learned_sparse | chem_gap | Winner |
|---|---:|---:|---:|---:|---:|---|
| `route` | 50.88% | **78.22%** | 73.05% | 77.15% | 77.15% | fixed_random |
| `order` | **51.76%** | 51.17% | 50.29% | 50.29% | 51.17% | none/random |
| `kv` | 53.03% | 67.09% | 67.38% | **67.97%** | 58.50% | learned_sparse |

### Interpretation / Интерпретация

1. **Connections matter.**
   `none` stays near random, while sparse graphs solve `route` and improve `kv`.

   **Связи реально важны.** Без связей motor-узлы почти не получают сенсорную информацию.

2. **Dense all-to-all is not automatically better.**
   Sparse/fixed graph beats dense on `route` and is competitive on `kv`.

   **Плотная матрица не всегда лучше.** Структура связности важнее, чем просто “всё со всем”.

3. **Plain scalar edges are not enough for temporal order.**
   All graph-only variants stay near random on `order`.

   **Обычных весов-связей мало для порядка событий.** Нужен специальный тип связи/оператор времени.

---

## Experiment 2: graph + typed operator channels

File:

```text
adfc/graph_adfc_worm_plus.py
```

Adds optional typed channels:

```text
DirectionalOrderChannel: A_before_B - B_before_A
KeyedReadChannel: query-key reads shifted value
```

Smoke command:

```bash
python -u adfc/graph_adfc_worm_plus.py \
  --out results/graph_plus_smoke \
  --steps 40 \
  --batch 128 \
  --eval-batch 192 \
  --tasks order,kv \
  --variants none,learned_sparse,learned_sparse_ops
```

### Results

| Task | none | learned_sparse | learned_sparse_ops | Winner |
|---|---:|---:|---:|---|
| `order` | 50.52% | 50.65% | **52.34%** | weak improvement |
| `kv` | 50.00% | 52.86% | **69.66%** | learned_sparse_ops |

### Interpretation / Интерпретация

Typed key-read helps `kv` immediately:

```text
kv: 50.00% -> 69.66%
```

But the learned order detector did not yet lock in:

```text
order_abs ≈ 0.004
```

So the next version should use a stronger relation bank:

```text
Pairwise sensory relation bank:
  for every sensory channel pair (i, j):
      i_before_j - j_before_i
```

This is closer to the conclusion from ADFC6: order is not just a weight, it is a relation operator.

---

## Current conclusion / Текущий вывод

The first graph experiments support a layered hypothesis:

```text
1. Cell dynamics matter.
2. Connectivity matters.
3. Connectivity topology matters.
4. For some tasks, edge TYPES/operators matter more than scalar weights.
```

Biological analogy:

```text
C. elegans is not one connection matrix.
It has chemical synapses, gap junctions, neuromodulation, body feedback, and cell-specific dynamics.
```

So a useful artificial worm-like architecture should probably be:

```text
GraphADFC = active cells + trainable sparse connections + typed relation channels
```

Not just:

```text
h_next = W @ h
```

---

## Next patch / Следующий патч

Recommended next experiment:

```text
GraphADFC-Worm-TypedBank
```

Add:

```text
PairwiseOrderBank over sensory channels
KeyValueBank over sensory channels
Sparse graph communication
Optional gap/diffusion
```

Expected behavior:

| Task | Needed mechanism |
|---|---|
| `route` | sparse graph communication |
| `kv` | graph + keyed read |
| `order` | pairwise directional relation bank |


---

## Experiment 3: TypedBank relation operators

File:

```text
adfc/graph_adfc_worm_typed_bank.py
```

This version replaces the weak learned order detector with a stronger general relation bank:

```text
PairwiseOrderBank:
  for every sensory channel pair (i, j):
      i_before_j - j_before_i
```

It also keeps:

```text
KeyReadBank:
  query-conditioned read over shifted values
```

Command:

```bash
python -u adfc/graph_adfc_worm_typed_bank.py \
  --out results/graph_typed_bank_full \
  --steps 100 \
  --batch 192 \
  --eval-batch 256 \
  --tasks route,order,kv \
  --variants none,learned_sparse,learned_sparse_typed \
  --log-every 25
```

### Full results

| Task | none | learned_sparse | learned_sparse_typed | Winner |
|---|---:|---:|---:|---|
| `route` | 51.86% | 76.07% | **100.00%** | learned_sparse_typed |
| `order` | 51.86% | 52.54% | **99.90%** | learned_sparse_typed |
| `kv` | 51.07% | 66.60% | **99.90%** | learned_sparse_typed |

### What changed

Previous typed attempt:

```text
learned DirectionalOrderChannel
order_abs ≈ 0.004
order acc ≈ 52.34%
```

New typed bank:

```text
PairwiseOrderBank
order_abs ≈ 0.30
order acc ≈ 99.90%
```

So the failure was not that typed edges are useless. The failure was that the first order edge detector did not expose the relation clearly enough.

### Main interpretation

This strongly supports the layered view:

```text
scalar graph edges        -> enough for simple routing / partial recall
relation typed edges      -> needed for temporal/order computation
keyed typed read edges    -> useful for associative recall
```

In other words:

```text
connections are not one thing.
there are different species of connection.
```

A better artificial worm should therefore be closer to:

```text
cell state
+ sparse chemical graph
+ gap/diffusion graph
+ pairwise relation bank
+ keyed memory read bank
+ learned routing over edge types
```

Not just:

```text
one dense learned matrix
```

### Caveat

This is still a toy benchmark. It proves the mechanism inside controlled synthetic tasks, not biological realism and not general real-world performance yet.

The next useful step is to make the edge-type router learn when to use:

```text
chemical / sparse graph
pairwise order relation
keyed read
gap diffusion
```

instead of always adding all typed channels.
