# 012 — Message-utility Task-MoE

Goal: improve 011 by adding best checkpoint saving and stronger graph utility.

## Files

- Code: `adfc/graph_adfc_worm_task_moe_msgutil.py`
- Runner: `run_012_msgutil_check.sh`
- Smoke: `results/012_smoke/`
- Full output: `results/012_task_moe_msgutil_s160/`

## What changed vs 011

011 already showed clear task/channel specialization and reached 96.78% peak, but graph utility stayed zero.

012 changes:

- default steps = 160, because 011 peaked at step 160 and degraded by step 200
- saves `best_<model>.pt` checkpoint when validation improves
- graph utility uses message-gradient signal instead of only task gate gradient
- structural update receives actual mean task probabilities from the current batch

## Utility approximation

During graph forward, the model stores graph messages and attention-like edge weights. After backward, utility is estimated from message influence:

```text
receiver_utility = abs(message * grad(message))
edge_utility ~= edge_weight * receiver_utility
```

This is still approximate, but closer to true edge usefulness than the older gate-gradient proxy.

## Command

```bash
./run_012_msgutil_check.sh
```

It compares only:

- `always_typed`
- `task_moe_msgutil`

## Main metrics

Inspect:

- `best_val_acc`
- `acc_program`
- `taskp_*_*`
- `w_graph_*`, `w_order_*`, `w_key_*`
- `graph_gate_route/order/kv/program`
- `edge_util_route/order/kv/program`
- saved `best_*.pt`

## Smoke

Smoke passed on Tesla P40. It confirms the retained message gradients do not break runtime.
