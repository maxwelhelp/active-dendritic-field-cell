#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-ADFC-Worm prototype.

Question tested:
  Are connections/edges the real computational substrate?

We keep the same recurrent cell and change only graph wiring variants:
  none            : no sensory->motor communication
  fixed_random    : fixed random sparse chemical edges
  learned_dense   : learned dense directed chemical graph
  learned_sparse  : learned weights on fixed sparse candidate graph
  chem_gap        : learned chemical + learned symmetric gap-junction-like diffusion + global modulator

Tasks:
  route : mode selects A-value or B-value
  order : decide whether A event happened before B, with optional mode flip
  kv    : 4-key associative recall

This is NOT a biological C. elegans simulation yet. It is a first GraphADFC
lab for testing whether trainable connections matter.
"""
from __future__ import annotations

import argparse, csv, json, math, random, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this experiment is GPU-only")
    return torch.device("cuda")


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nparams(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def wcsv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader(); w.writerows(rows)


# ------------------------------- tasks -------------------------------

@torch.no_grad()
def make_route(B: int, T: int, Fdim: int, dev) -> Tuple[torch.Tensor, torch.Tensor]:
    x = 0.02 * torch.randn(B, T, Fdim, device=dev)
    rows = torch.arange(B, device=dev)
    mode = torch.randint(0, 2, (B,), device=dev)
    va = torch.randint(0, 2, (B,), device=dev)
    vb = torch.randint(0, 2, (B,), device=dev)
    pa = torch.randint(5, T // 2 - 3, (B,), device=dev)
    pb = torch.randint(T // 2, T - 5, (B,), device=dev)
    x[:, 0, 0] = 1.0 - mode.float()
    x[:, 0, 1] = mode.float()
    x[rows, pa, 2] = 1.0              # A marker
    x[rows, pa + 1, 4 + va] = 1.0     # A value bit via channels 4/5
    x[rows, pb, 3] = 1.0              # B marker
    x[rows, pb + 1, 4 + vb] = 1.0     # B value bit
    x[:, -1, 12] = 1.0                # query
    y = torch.where(mode == 0, va, vb).long()
    return x, y


@torch.no_grad()
def make_order(B: int, T: int, Fdim: int, dev) -> Tuple[torch.Tensor, torch.Tensor]:
    x = 0.02 * torch.randn(B, T, Fdim, device=dev)
    rows = torch.arange(B, device=dev)
    mode = torch.randint(0, 2, (B,), device=dev)
    p1 = torch.randint(4, T - 8, (B,), device=dev)
    p2 = (p1 + torch.randint(2, 16, (B,), device=dev)).clamp(max=T - 4)
    swap = torch.randint(0, 2, (B,), device=dev).bool()
    pa = torch.where(swap, p2, p1)
    pb = torch.where(swap, p1, p2)
    x[:, 0, 0] = 1.0 - mode.float()
    x[:, 0, 1] = mode.float()
    x[rows, pa, 6] = 1.0              # event A
    x[rows, pb, 7] = 1.0              # event B
    x[:, -1, 12] = 1.0
    y = ((pa < pb).long() + mode).remainder(2).long()
    return x, y


@torch.no_grad()
def make_kv(B: int, T: int, Fdim: int, dev) -> Tuple[torch.Tensor, torch.Tensor]:
    x = 0.02 * torch.randn(B, T, Fdim, device=dev)
    rows = torch.arange(B, device=dev)
    vals = torch.randint(0, 2, (B, 4), device=dev)
    base = torch.tensor([6, 16, 26, 36], device=dev)
    pos = (base[None, :] + torch.randint(-2, 3, (B, 4), device=dev)).clamp(3, T - 6)
    for k in range(4):
        pk = pos[:, k]
        x[rows, pk, 8 + k] = 1.0          # key k
        x[rows, pk + 1, 4 + vals[:, k]] = 1.0  # value bit follows key
    q = torch.randint(0, 4, (B,), device=dev)
    x[:, -2, 12] = 1.0
    x[rows, T - 1, 8 + q] = 1.0          # query key
    y = vals[rows, q].long()
    return x, y


TASKS = {"route": make_route, "order": make_order, "kv": make_kv}


# ------------------------------- model -------------------------------

class SharedADFCCell(nn.Module):
    def __init__(self, d: int, n_nodes: int | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, 3 * d)
        self.msg = nn.Linear(d, 3 * d, bias=False)
        self.decay = nn.Parameter(torch.full((n_nodes,), 0.55) if n_nodes is not None else torch.tensor(0.55))

    def forward(self, h: torch.Tensor, u: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        z = self.norm(h)
        a = self.inp(u) + self.msg(m)
        gate, cand, plateau = a.chunk(3, dim=-1)
        gate = torch.sigmoid(gate)
        plateau = torch.sigmoid(plateau)
        cand = torch.tanh(cand + 0.25 * z)
        slow = torch.sigmoid(self.decay)
        if slow.ndim == 1:
            slow = slow[None, :, None]
        new = slow * h + (1.0 - slow) * (plateau * cand)
        return (1.0 - gate) * h + gate * new


class GraphADFCWorm(nn.Module):
    def __init__(self, variant: str, n_nodes: int = 48, d: int = 48, fdim: int = 16,
                 sensor_nodes: int = 16, motor_nodes: int = 4, candidate_degree: int = 6):
        super().__init__()
        self.variant = variant
        self.n_nodes = n_nodes
        self.d = d
        self.fdim = fdim
        self.sensor_nodes = sensor_nodes
        self.motor_nodes = motor_nodes
        self.candidate_degree = candidate_degree

        self.sensor_embed = nn.Parameter(torch.randn(fdim, d) / math.sqrt(d))
        self.node_bias = nn.Parameter(torch.zeros(n_nodes, d))
        self.cell = SharedADFCCell(d)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d * motor_nodes, 2)
        self.mod = nn.Sequential(nn.Linear(d, d), nn.Tanh())

        # Directed chemical edges: dest i receives from src j.
        if variant in {"learned_dense", "chem_gap"}:
            self.A_logits = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.02)
        elif variant == "learned_sparse":
            self.A_logits = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.02)
        else:
            self.register_parameter("A_logits", None)

        mask = torch.zeros(n_nodes, n_nodes)
        rng = random.Random(123)
        # ensure paths from sensors -> hidden -> motors exist in sparse/fixed variants
        hidden_start = sensor_nodes
        motor_start = n_nodes - motor_nodes
        for i in range(n_nodes):
            choices = list(range(n_nodes))
            if i in choices:
                choices.remove(i)
            for j in rng.sample(choices, min(candidate_degree, len(choices))):
                mask[i, j] = 1.0
        for dst in range(hidden_start, n_nodes):
            for src in range(sensor_nodes):
                if rng.random() < 0.22:
                    mask[dst, src] = 1.0
        for dst in range(motor_start, n_nodes):
            for src in range(hidden_start, motor_start):
                if rng.random() < 0.35:
                    mask[dst, src] = 1.0
        mask.fill_diagonal_(0.0)
        self.register_buffer("candidate_mask", mask)

        fixed = torch.rand(n_nodes, n_nodes) * mask
        fixed = fixed / fixed.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("fixed_A", fixed)

        if variant == "chem_gap":
            self.G_logits = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.02)
            self.gap_scale = nn.Parameter(torch.tensor(0.15))
            self.chem_scale = nn.Parameter(torch.tensor(1.0))
            self.mod_scale = nn.Parameter(torch.tensor(0.25))
        else:
            self.register_parameter("G_logits", None)
            self.register_parameter("gap_scale", None)
            self.register_parameter("chem_scale", None)
            self.register_parameter("mod_scale", None)

        self.last_stats: Dict[str, float] = {}

    def chemical_A(self) -> torch.Tensor:
        N = self.n_nodes
        if self.variant == "none":
            return torch.zeros(N, N, device=self.node_bias.device)
        if self.variant == "fixed_random":
            return self.fixed_A
        if self.variant == "learned_dense" or self.variant == "chem_gap":
            logits = self.A_logits.masked_fill(torch.eye(N, device=self.A_logits.device).bool(), -1e4)
            return torch.softmax(logits, dim=1)
        if self.variant == "learned_sparse":
            w = torch.sigmoid(self.A_logits) * self.candidate_mask
            return w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        raise ValueError(self.variant)

    def gap_G(self) -> torch.Tensor:
        if self.variant != "chem_gap":
            return torch.zeros(self.n_nodes, self.n_nodes, device=self.node_bias.device)
        g = torch.sigmoid(self.G_logits)
        g = 0.5 * (g + g.t())
        g = g * (1.0 - torch.eye(self.n_nodes, device=g.device))
        g = g / g.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, Fdim = x.shape
        N, D = self.n_nodes, self.d
        h = self.node_bias.unsqueeze(0).expand(B, N, D).contiguous()
        A = self.chemical_A()
        G = self.gap_G()
        motor_start = N - self.motor_nodes

        for t in range(T):
            u = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
            # channel c injects into sensor node c
            inj = x[:, t, :self.sensor_nodes].unsqueeze(-1) * self.sensor_embed[:self.sensor_nodes].unsqueeze(0)
            u[:, :self.sensor_nodes, :] = inj

            chem = torch.einsum("ij,bjd->bid", A, h)
            if self.variant == "chem_gap":
                avg = torch.einsum("ij,bjd->bid", G, h)
                gap = avg - h
                global_mod = self.mod(h[:, :self.sensor_nodes].mean(dim=1)).unsqueeze(1)
                msg = self.chem_scale.tanh() * chem + self.gap_scale.tanh() * gap + self.mod_scale.tanh() * global_mod
            else:
                msg = chem
            h = self.cell(h, u, msg)

        z = self.out_norm(h[:, motor_start:, :]).reshape(B, -1)
        with torch.no_grad():
            nonzero = float((A.detach() > 1e-4).float().mean().cpu())
            ent = float((-(A.clamp_min(1e-9) * A.clamp_min(1e-9).log()).sum(dim=1).mean() / math.log(max(2, N))).detach().cpu()) if self.variant != "none" else 0.0
            self.last_stats = {"edge_density": nonzero, "edge_entropy": ent}
            if self.variant == "chem_gap":
                self.last_stats.update({
                    "chem_scale": float(self.chem_scale.detach().tanh().cpu()),
                    "gap_scale": float(self.gap_scale.detach().tanh().cpu()),
                    "mod_scale": float(self.mod_scale.detach().tanh().cpu()),
                })
        return self.head(z)

    def stats(self):
        return dict(self.last_stats)


# ------------------------------- train/eval -------------------------------

@torch.no_grad()
def evaluate(model, task_name, args, dev, nb=4):
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


def train_one(variant, task_name, args, dev):
    model = GraphADFCWorm(variant, args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree).to(dev)
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
            vl, va = evaluate(model, task_name, args, dev, nb=4)
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
            print(f"[{task_name}] {variant:14s} {step:04d}/{args.steps} loss={row['train_loss']:.4f} val={100*va:6.2f}% best={100*best['acc']:6.2f}% edges={row.get('edge_density',0):.3f} ent={row.get('edge_entropy',0):.3f}", flush=True)
    return rows, rows[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/graph_adfc_worm")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=100)
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
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--tasks", default="route,order,kv")
    p.add_argument("--variants", default="none,fixed_random,learned_dense,learned_sparse,chem_gap")
    args = p.parse_args()
    dev = require_cuda()
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("=== Graph-ADFC-Worm connection benchmark ===", flush=True)
    print("gpu", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    print("config", json.dumps(vars(args), ensure_ascii=False), flush=True)

    all_rows, finals = [], []
    tasks = [x for x in args.tasks.split(",") if x]
    variants = [x for x in args.variants.split(",") if x]
    for task in tasks:
        print("\n--- TASK", task, "---", flush=True)
        for variant in variants:
            rows, fin = train_one(variant, task, args, dev)
            all_rows += rows; finals.append(fin)
            wcsv(out / "train_rows.csv", all_rows)
            wcsv(out / "summary_rows.csv", finals)
    winners = []
    for task in tasks:
        sub = [r for r in finals if r["task"] == task]
        win = max(sub, key=lambda r: r["best_val_acc"])
        none = next((r for r in sub if r["variant"] == "none"), None)
        winners.append({
            "task": task,
            "winner": win["variant"],
            "winner_acc": win["best_val_acc"],
            "none_acc": none["best_val_acc"] if none else None,
            "winner_margin_vs_none": win["best_val_acc"] - (none["best_val_acc"] if none else 0.0),
        })
    wcsv(out / "winners.csv", winners)
    (out / "summary.json").write_text(json.dumps({"finals": finals, "winners": winners}, indent=2), encoding="utf-8")
    print("\n=== WINNERS ===", flush=True)
    for w in winners:
        print(f"{w['task']} winner={w['winner']} acc={100*w['winner_acc']:.2f}% margin_vs_none={100*w['winner_margin_vs_none']:+.2f}%", flush=True)
    print("DONE", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
