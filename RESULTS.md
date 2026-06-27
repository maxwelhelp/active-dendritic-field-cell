# Results / Результаты

All runs below were executed on:

```text
GPU: NVIDIA Tesla P40
PyTorch: 2.6.0+cu124
```

Все прогоны ниже выполнялись на:

```text
GPU: NVIDIA Tesla P40
PyTorch: 2.6.0+cu124
```

---

## Final repository benchmark / Финальный бенчмарк репозитория

Run directory:

```text
results/final_gpu_bench/
```

Command:

```bash
python -u adfc/adfc_v6_orderkernel.py \
  --out results/final_gpu_bench \
  --steps 120 \
  --batch 192 \
  --eval-batch 256 \
  --tasks mode_select,order_compare,kv_recall4 \
  --models mean_pool,adfc3,adfc6
```

### Best validation accuracy

| Task | mean_pool | ADFC3 | ADFC6 | Comment |
|---|---:|---:|---:|---|
| `mode_select` | 75.59% | **100.00%** | **100.00%** | mode/conditional memory solved |
| `order_compare` | 51.56% | 61.62% | **67.48%** | improves at 120 steps; longer run reaches 99.85% |
| `kv_recall4` | 68.85% | **100.00%** | **100.00%** | key-value recall solved |

### Русская интерпретация

| Задача | mean_pool | ADFC3 | ADFC6 | Комментарий |
|---|---:|---:|---:|---|
| `mode_select` | 75.59% | **100.00%** | **100.00%** | условный режим/память решены |
| `order_compare` | 51.56% | 61.62% | **67.48%** | за 120 шагов уже лучше; отдельный 180-step прогон даёт 99.85% |
| `kv_recall4` | 68.85% | **100.00%** | **100.00%** | key-value память решена |

---

## Longer order-only run / Длинный прогон только порядка

Run directory:

```text
results/v6_order/
```

This run used 180 steps on `order_compare` and shows why the directional order kernel matters.

Этот прогон использовал 180 шагов на `order_compare` и показывает, зачем нужен directional order kernel.

| Model | Best validation accuracy |
|---|---:|
| mean_pool | 51.37% |
| ADFC3 | 86.96% |
| **ADFC6** | **99.85%** |

Key training curve for ADFC6:

```text
step  90: 58.84%
step 120: 86.67%
step 150: 98.54%
step 180: 99.85%
```

Interpretation:

- The order kernel needs enough training steps to become useful.
- Once it locks in, the directional relation `A_before_B - B_before_A` solves the temporal order task almost perfectly.

Интерпретация:

- order-kernel требует достаточно шагов, чтобы включиться.
- Когда он “схватывает” детекторы событий, отношение `A_before_B - B_before_A` почти идеально решает задачу порядка.

---

## Earlier targeted ADFC3 run / Ранний точечный прогон ADFC3

Run directory:

```text
results/v3_targeted/
```

| Task | mean_pool | ADFC3 | Margin |
|---|---:|---:|---:|
| `mode_select` | 75.78% | **100.00%** | +24.22% |
| `order_compare` | 51.66% | 54.20% | +2.54% |
| `kv_recall4` | 71.48% | **100.00%** | +28.52% |

This showed that keyed memory solved associative recall, but temporal order was still weak.

Это показало, что keyed memory решает ассоциативное извлечение, но порядок событий ещё слабый.

---

## Main conclusion / Главный вывод

English:

```text
ADFC3 = dendritic state + keyed memory
  solves: mode_select, kv_recall4
  weak on: order_compare

ADFC6 = ADFC3 + directional order kernel
  solves: mode_select, kv_recall4, order_compare with enough training
```

Русский:

```text
ADFC3 = дендритное состояние + keyed memory
  решает: mode_select, kv_recall4
  слабое место: order_compare

ADFC6 = ADFC3 + directional order kernel
  решает: mode_select, kv_recall4, order_compare при достаточном числе шагов
```

The architecture is still synthetic-task-only. The next required step is comparing against GRU, tiny attention, linear attention, and real sequence tasks.

Архитектура пока проверена только на синтетике. Следующий обязательный шаг — сравнение с GRU, tiny attention, linear attention и реальные sequence-задачи.
