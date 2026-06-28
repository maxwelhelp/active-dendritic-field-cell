# 015 — Real GPU TaskMoE/ADFC Life

This replaces the earlier CPU/Numpy toy brain with the real imported project network.

Real imports used by the policy:

```python
from graph_adfc_worm import require_cuda, nparams, wcsv
from graph_adfc_worm_typed_bank import PairwiseOrderBank, KeyReadBank
from genome_task_moe import GraphChannel, TaskMoEDNA, AlwaysTypedDNA
```

Policy branches:

```text
GraphChannel -> graph action head
PairwiseOrderBank -> order action head
KeyReadBank -> key action head
Task router -> channel mixture
```

## Fixed bugs

Two autograd bugs were fixed:

1. Environment tensors were modified before backward.
2. Environment energy accidentally kept graph history because cost used the non-detached action tensor.

The current loop is:

```text
build obs on CUDA
forward real TaskMoE/ADFC policy
compute MSE policy loss
loss.backward()
optimizer.step()
use detached action for world physics
detach environment state before next step
```

## Verified run

Command:

```bash
TORCH_DISABLE_ADDR2LINE=1 /home/maxwelhelp/main/bin/python -u adfc/life_task_moe_gpu.py \
  --out results/015_verify_200 \
  --steps 200 \
  --agents 256 \
  --food 64 \
  --log-every 20
```

Result: completed 200 steps on Tesla P40.

Loss trend:

```text
step 1   loss 0.1334
step 20  loss 0.0549
step 100 loss 0.0614
step 160 loss 0.0490
step 200 loss 0.0607
```

## Visual runner

```bash
./run_015_life_task_moe_gpu_visual.sh
```

The visual wrapper imports and uses the same `LifeGPUEnv` and `LifeTaskMoEPolicy`; it does not define a separate toy brain.
