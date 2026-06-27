#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph-ADFC-Worm Budgeted Top-k Router.

Experiment 008.

Adds three anti-collapse mechanisms on top of experiment 007:
  1) top-k channel routing: each sample can use only k channels;
  2) channel dropout: during training, typed channels are sometimes unavailable;
  3) target/load loss: encourage expected specialization by task type.

Channels:
  graph = sparse graph communication
  order = PairwiseOrderBank
  key   = KeyReadBank

Expected specialization:
  route   -> graph mostly
  order   -> order mostly
  kv      -> key mostly
  program -> order + key, with maybe small graph
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_adfc_worm import GraphADFCWorm, require_cuda, seed_all, nparams, wcsv
from graph_adfc_worm_cost_router import (
    ALL_TASKS,
    TASK_NAMES,
    AlwaysTyped,
    CostAwareRouter,
    make_mixed,
)


CHANNELS = ["graph", "order", "key"]
TARGETS = {
    "route":   [0.75, 0.10, 0.15],
    "order":   [0.05, 0.90, 0.05],
    "kv":      [0.05, 0.15, 0.80],
    "program": [0.05, 0.45, 0.50],
}


class BudgetedTopKRouter(CostAwareRouter):
    def __init__(self, *args, top_k: int = 2, channel_dropout: float = 0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = int(top_k)
        self.channel_dropout = float(channel_dropout)

    def _apply_channel_dropout(self, logits: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.channel_dropout <= 0.0:
            return logits
        keep = torch.rand_like(logits).ge(self.channel_dropout)
        # Never drop all channels for a sample.
        all_dropped = ~keep.any(dim=-1)
        if all_dropped.any():
            keep[all_dropped, torch.randint(0, logits.shape[-1], (int(all_dropped.sum()),), device=logits.device)] = True
        return logits.masked_fill(~keep, -1e4)

    def _apply_topk(self, logits: torch.Tensor) -> torch.Tensor:
        k = max(1, min(self.top_k, logits.shape[-1]))
        vals, idx = torch.topk(logits, k=k, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, value=True)
        return logits.masked_fill(~mask, -1e4)

    def route_weights(self, x: torch.Tensor):
        feat = torch.cat([x.mean(1), x.max(1).values, x[:, -1]], dim=-1)
        logits = self.router(feat)
        routed_logits = self._apply_topk(self._apply_channel_dropout(logits))
        soft = torch.softmax(routed_logits, dim=-1)
        if self.hard and self.training:
            idx = soft.argmax(dim=-1)
            hard = F.one_hot(idx, num_classes=3).to(soft.dtype)
            return hard + soft - soft.detach(), logits
        return soft, logits


def target_for_task_ids(task_ids: torch.Tensor, eval_tasks: list[str], dev) -> torch.Tensor:
    rows = []
    for name in eval_tasks:
        rows.append(TARGETS.get(name, [1.0 / 3, 1.0 / 3, 1.0 / 3]))
    table = torch.tensor(rows, dtype=torch.float32, device=dev)
    return table[task_ids]


def build_model(name: str, args):
    if name == "graph":
        return GraphADFCWorm("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree)
    if name == "always_typed":
        return AlwaysTyped("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree)
    if name == "cost_router":
        return CostAwareRouter("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, hard=False)
    if name == "budget_topk":
        return BudgetedTopKRouter(
            "learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree,
            hard=False, top_k=args.top_k, channel_dropout=args.channel_dropout,
        )
    if name == "budget_topk_hard":
        return BudgetedTopKRouter(
            "learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree,
            hard=True, top_k=args.top_k, channel_dropout=args.channel_dropout,
        )
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, task_name: str, args, dev, eval_tasks: list[str], nb: int = 4):
    model.eval()
    correct = total = 0
    losses = []
    per = {n: [0, 0] for n in eval_tasks}
    wsum = {n: torch.zeros(3, device=dev) for n in eval_tasks}

    for _ in range(nb):
        if task_name == "mixed":
            x, y, tid = make_mixed(args.eval_batch, args.seq_len, args.fdim, dev, eval_tasks)
        else:
            x, y = ALL_TASKS[task_name](args.eval_batch, args.seq_len, args.fdim, dev)
            tid = torch.zeros(args.eval_batch, device=dev, dtype=torch.long)

        logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        losses.append(float(loss.cpu()))

        if hasattr(model, "route_weights"):
            weights, _ = model.route_weights(x)
        else:
            st = model.stats()
            weights = torch.tensor(
                [st.get("w_graph", 0.0), st.get("w_order", 0.0), st.get("w_key", 0.0)],
                device=dev,
            ).view(1, 3).expand(x.shape[0], 3)

        if task_name == "mixed":
            for i, n in enumerate(eval_tasks):
                mask = tid == i
                if mask.any():
                    per[n][0] += int((pred[mask] == y[mask]).sum().item())
                    per[n][1] += int(mask.sum().item())
                    wsum[n] += weights[mask].sum(0)
        else:
            n = task_name
            per[n][0] += int((pred == y).sum().item())
            per[n][1] += int(y.numel())
            wsum[n] += weights.sum(0)

    out = {"val_loss": sum(losses) / len(losses), "val_acc": correct / max(1, total)}
    for n in eval_tasks:
        if per[n][1] > 0:
            out[f"acc_{n}"] = per[n][0] / per[n][1]
            w = wsum[n] / per[n][1]
            out[f"w_graph_{n}"] = float(w[0].cpu())
            out[f"w_order_{n}"] = float(w[1].cpu())
            out[f"w_key_{n}"] = float(w[2].cpu())
    return out


def train_one(model_name: str, task_name: str, args, dev, eval_tasks: list[str]):
    model = build_model(model_name, args).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    costs = torch.tensor([args.cost_graph, args.cost_order, args.cost_key], device=dev)
    rows = []
    best = {"acc": 0.0, "step": 0}
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        if task_name == "mixed":
            x, y, tid = make_mixed(args.batch, args.seq_len, args.fdim, dev, eval_tasks)
        else:
            x, y = ALL_TASKS[task_name](args.batch, args.seq_len, args.fdim, dev)
            tid = torch.zeros(args.batch, device=dev, dtype=torch.long)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        ce = F.cross_entropy(logits.float(), y)
        loss = ce
        exp_cost = torch.tensor(0.0, device=dev)
        entropy = torch.tensor(0.0, device=dev)
        target_loss = torch.tensor(0.0, device=dev)
        global_usage_loss = torch.tensor(0.0, device=dev)

        if hasattr(model, "route_weights"):
            weights, _ = model.route_weights(x)
            exp_cost = (weights * costs.view(1, 3)).sum(-1).mean()
            entropy = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(-1).mean() / math.log(3)
            target = target_for_task_ids(tid, eval_tasks if task_name == "mixed" else [task_name], dev)
            target_loss = F.mse_loss(weights, target)
            global_usage_loss = F.mse_loss(weights.mean(0), target.mean(0))
            loss = (
                ce
                + args.cost_lambda * exp_cost
                + args.entropy_penalty * entropy
                + args.target_lambda * target_loss
                + args.balance_lambda * global_usage_loss
            )

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            ev = evaluate(model, task_name, args, dev, eval_tasks, 4)
            if ev["val_acc"] > best["acc"]:
                best = {"acc": ev["val_acc"], "step": step}
            row = {
                "task": task_name,
                "model": model_name,
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "train_ce": float(ce.detach().cpu()),
                "expected_cost": float(exp_cost.detach().cpu()),
                "entropy_loss": float(entropy.detach().cpu()),
                "target_loss": float(target_loss.detach().cpu()),
                "global_usage_loss": float(global_usage_loss.detach().cpu()),
                "val_loss": ev["val_loss"],
                "val_acc": ev["val_acc"],
                "best_val_acc": best["acc"],
                "best_step": best["step"],
                "params": nparams(model),
                "grad_norm": float(gn.detach().cpu()),
                "sec": time.time() - t0,
            }
            row.update(model.stats() if hasattr(model, "stats") else {})
            row.update(ev)
            rows.append(row)
            extra = " ".join([f"{n}={100 * ev.get('acc_' + n, 0):.1f}" for n in eval_tasks])
            print(
                f"[{task_name}] {model_name:16s} {step:04d}/{args.steps} "
                f"val={100 * ev['val_acc']:6.2f}% best={100 * best['acc']:6.2f}% "
                f"cost={row['expected_cost']:.3f} tgt={row['target_loss']:.3f} "
                f"wg={row.get('w_graph', 0):.2f} wo={row.get('w_order', 0):.2f} wk={row.get('w_key', 0):.2f} {extra}",
                flush=True,
            )
    return rows, rows[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/008_budgeted_topk")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=56)
    p.add_argument("--fdim", type=int, default=16)
    p.add_argument("--nodes", type=int, default=48)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--sensor-nodes", type=int, default=16)
    p.add_argument("--motor-nodes", type=int, default=4)
    p.add_argument("--degree", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cost-graph", type=float, default=0.05)
    p.add_argument("--cost-order", type=float, default=0.55)
    p.add_argument("--cost-key", type=float, default=0.35)
    p.add_argument("--cost-lambda", type=float, default=0.02)
    p.add_argument("--entropy-penalty", type=float, default=0.005)
    p.add_argument("--target-lambda", type=float, default=0.08)
    p.add_argument("--balance-lambda", type=float, default=0.04)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--channel-dropout", type=float, default=0.12)
    p.add_argument("--log-every", type=int, default=40)
    p.add_argument("--tasks", default="mixed")
    p.add_argument("--mixed-tasks", default="route,order,kv,program")
    p.add_argument("--models", default="graph,always_typed,cost_router,budget_topk,budget_topk_hard")
    args = p.parse_args()

    dev = require_cuda()
    seed_all(args.seed)
    eval_tasks = [x for x in args.mixed_tasks.split(",") if x]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("=== Graph-ADFC-Worm budgeted top-k router benchmark ===", flush=True)
    print("gpu", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    print("targets", json.dumps(TARGETS, indent=2), flush=True)

    all_rows = []
    finals = []
    for task in [x for x in args.tasks.split(",") if x]:
        print("\n--- TASK", task, "---", flush=True)
        for model_name in [x for x in args.models.split(",") if x]:
            rows, fin = train_one(model_name, task, args, dev, eval_tasks)
            all_rows += rows
            finals.append(fin)
            wcsv(out / "train_rows.csv", all_rows)
            wcsv(out / "summary_rows.csv", finals)

    winners = []
    for task in [x for x in args.tasks.split(",") if x]:
        sub = [r for r in finals if r["task"] == task]
        win = max(sub, key=lambda r: r["best_val_acc"])
        winners.append({"task": task, "winner": win["model"], "winner_acc": win["best_val_acc"]})
    wcsv(out / "winners.csv", winners)
    (out / "summary.json").write_text(json.dumps({"finals": finals, "winners": winners}, indent=2), encoding="utf-8")

    print("\n=== WINNERS ===", flush=True)
    for w in winners:
        print(f"{w['task']} winner={w['winner']} acc={100 * w['winner_acc']:.2f}%", flush=True)
    print("DONE", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
