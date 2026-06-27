#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph-ADFC-Worm TypedBank.

Checks the idea that intelligence is not only in cell state or scalar edge weights,
but also in typed connection operators:
  - sparse graph edges carry state between nodes
  - pairwise order bank computes channel_i_before_channel_j relations
  - key read bank computes query-conditioned recall
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_adfc_worm import TASKS, GraphADFCWorm, require_cuda, seed_all, nparams, wcsv


class PairwiseOrderBank(nn.Module):
    def __init__(self, fdim: int, d: int, hidden: int = 96):
        super().__init__()
        self.fdim = fdim
        self.net = nn.Sequential(
            nn.LayerNorm(fdim * fdim),
            nn.Linear(fdim * fdim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.Tanh(),
        )
        self.last_abs = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]. Build all directional temporal relations.
        s = x.clamp_min(0.0)
        prefix = torch.cumsum(s, dim=1) - s
        # before[i,j] = amount of channel i before channel j.
        before = torch.einsum("bti,btj->bij", prefix, s)
        antisym = before - before.transpose(1, 2)
        denom = (s.sum(1).unsqueeze(2) * s.sum(1).unsqueeze(1)).clamp_min(1e-4)
        rel = antisym / denom
        self.last_abs = rel.detach().abs().mean()
        return self.net(rel.reshape(x.shape[0], -1))


class KeyReadBank(nn.Module):
    def __init__(self, fdim: int, d: int):
        super().__init__()
        self.q = nn.Linear(fdim, d, bias=False)
        self.k = nn.Linear(fdim, d, bias=False)
        self.v = nn.Linear(fdim, d, bias=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.Tanh())
        self.scale = d ** -0.5
        self.last_entropy = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Query is encoded in the final two tokens. Value is shifted one step after key.
        keys = self.k(x)
        values = self.v(torch.cat([x[:, 1:], x[:, -1:]], dim=1))
        q = self.q(x[:, -1] + x[:, -2])
        logits = torch.einsum("bd,btd->bt", q, keys) * self.scale
        att = torch.softmax(logits, dim=1)
        read = torch.einsum("bt,btd->bd", att, values)
        self.last_entropy = (-(att * att.clamp_min(1e-9).log()).sum(1).mean() / math.log(x.shape[1])).detach()
        return self.out(read)


class GraphADFCWormTypedBank(nn.Module):
    def __init__(self, base_variant: str, n_nodes: int, d: int, fdim: int,
                 sensor_nodes: int, motor_nodes: int, degree: int, use_typed: bool):
        super().__init__()
        self.use_typed = use_typed
        self.base = GraphADFCWorm(base_variant, n_nodes, d, fdim, sensor_nodes, motor_nodes, degree)
        if use_typed:
            self.order_bank = PairwiseOrderBank(fdim, d)
            self.key_bank = KeyReadBank(fdim, d)
            self.typed_head = nn.Linear(2 * d, 2)
            self.typed_scale = nn.Parameter(torch.tensor(1.0))
        self.last_stats = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        stats = self.base.stats()
        if self.use_typed:
            order_z = self.order_bank(x)
            key_z = self.key_bank(x)
            typed_logits = self.typed_head(torch.cat([order_z, key_z], dim=-1))
            logits = logits + self.typed_scale.tanh() * typed_logits
            stats.update({
                "order_abs": float(self.order_bank.last_abs.cpu()),
                "key_entropy": float(self.key_bank.last_entropy.cpu()),
                "typed_scale": float(self.typed_scale.detach().tanh().cpu()),
            })
        self.last_stats = stats
        return logits

    def stats(self):
        return dict(self.last_stats)


def build_variant(name: str, args):
    use_typed = name.endswith("_typed")
    base = name.replace("_typed", "")
    return GraphADFCWormTypedBank(
        base, args.nodes, args.dim, args.fdim,
        args.sensor_nodes, args.motor_nodes, args.degree, use_typed
    )


@torch.no_grad()
def evaluate(model, task_name: str, args, dev, nb: int = 4):
    model.eval()
    make = TASKS[task_name]
    correct = total = 0
    losses = []
    for _ in range(nb):
        x, y = make(args.eval_batch, args.seq_len, args.fdim, dev)
        logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        correct += int((logits.argmax(-1) == y).sum().item())
        total += int(y.numel())
        losses.append(float(loss.cpu()))
    return sum(losses) / len(losses), correct / max(1, total)


def train_one(variant: str, task_name: str, args, dev):
    model = build_variant(variant, args).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    make = TASKS[task_name]
    rows = []
    best = {"acc": 0.0, "loss": 999.0, "step": 0}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x, y = make(args.batch, args.seq_len, args.fdim, dev)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            vl, va = evaluate(model, task_name, args, dev, 4)
            if va > best["acc"]:
                best = {"acc": va, "loss": vl, "step": step}
            row = {
                "task": task_name, "variant": variant, "step": step,
                "train_loss": float(loss.detach().cpu()), "val_loss": vl, "val_acc": va,
                "best_val_acc": best["acc"], "best_step": best["step"],
                "params": nparams(model), "grad_norm": float(gn.detach().cpu()), "sec": time.time() - t0,
            }
            row.update(model.stats())
            rows.append(row)
            print(f"[{task_name}] {variant:20s} {step:04d}/{args.steps} loss={row['train_loss']:.4f} val={100*va:6.2f}% best={100*best['acc']:6.2f}% ord={row.get('order_abs',0):.3f} ent={row.get('key_entropy',0):.3f}", flush=True)
    return rows, rows[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/graph_typed_bank")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--fdim", type=int, default=16)
    p.add_argument("--nodes", type=int, default=48)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--sensor-nodes", type=int, default=16)
    p.add_argument("--motor-nodes", type=int, default=4)
    p.add_argument("--degree", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--tasks", default="order,kv")
    p.add_argument("--variants", default="none,learned_sparse,learned_sparse_typed")
    args = p.parse_args()

    dev = require_cuda()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("=== Graph-ADFC-Worm TypedBank benchmark ===", flush=True)
    print("gpu", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)

    all_rows, finals = [], []
    tasks = [x for x in args.tasks.split(",") if x]
    variants = [x for x in args.variants.split(",") if x]
    for task in tasks:
        print("\n--- TASK", task, "---", flush=True)
        for variant in variants:
            rows, fin = train_one(variant, task, args, dev)
            all_rows += rows
            finals.append(fin)
            wcsv(out / "train_rows.csv", all_rows)
            wcsv(out / "summary_rows.csv", finals)

    winners = []
    for task in tasks:
        sub = [r for r in finals if r["task"] == task]
        win = max(sub, key=lambda r: r["best_val_acc"])
        none = next((r for r in sub if r["variant"] == "none"), None)
        base = none["best_val_acc"] if none else 0.0
        winners.append({
            "task": task,
            "winner": win["variant"],
            "winner_acc": win["best_val_acc"],
            "none_acc": base,
            "margin_vs_none": win["best_val_acc"] - base,
        })
    wcsv(out / "winners.csv", winners)
    (out / "summary.json").write_text(json.dumps({"finals": finals, "winners": winners}, indent=2), encoding="utf-8")
    print("\n=== WINNERS ===", flush=True)
    for w in winners:
        print(f"{w['task']} winner={w['winner']} acc={100*w['winner_acc']:.2f}% margin_vs_none={100*w['margin_vs_none']:+.2f}%", flush=True)
    print("DONE", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
