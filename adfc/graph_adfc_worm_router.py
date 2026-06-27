#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph-ADFC-Worm Router.

Goal:
  Test learned routing over connection/operator TYPES.

Channels:
  graph : sparse chemical graph communication
  order : pairwise temporal relation bank
  key   : query-conditioned key/value read

The router sees global input statistics and mixes channel logits per sample:
  logits = w_graph * graph_logits + w_order * order_logits + w_key * key_logits

Unlike graph_adfc_worm_typed_bank.py, this does not always add all typed channels
with one fixed scale. It learns which channel to use.
"""
from __future__ import annotations

import argparse, csv, json, math, random, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_adfc_worm import TASKS, GraphADFCWorm, require_cuda, seed_all, nparams, wcsv
from graph_adfc_worm_typed_bank import PairwiseOrderBank, KeyReadBank

TASK_NAMES = ["route", "order", "kv"]


@torch.no_grad()
def make_mixed(B: int, T: int, Fdim: int, dev):
    """Build one batch containing route/order/kv samples.

    Returns x, y, task_id where task_id is 0=route, 1=order, 2=kv.
    """
    chunks = []
    labels = []
    tids = []
    base = B // 3
    sizes = [base, base, B - 2 * base]
    for tid, (name, n) in enumerate(zip(TASK_NAMES, sizes)):
        x, y = TASKS[name](n, T, Fdim, dev)
        chunks.append(x); labels.append(y); tids.append(torch.full((n,), tid, device=dev, dtype=torch.long))
    x = torch.cat(chunks, dim=0)
    y = torch.cat(labels, dim=0)
    task_id = torch.cat(tids, dim=0)
    perm = torch.randperm(B, device=dev)
    return x[perm], y[perm], task_id[perm]


class RoutedGraphADFC(nn.Module):
    def __init__(self, base_variant: str, n_nodes: int, d: int, fdim: int,
                 sensor_nodes: int, motor_nodes: int, degree: int,
                 router_hidden: int = 64, hard: bool = False):
        super().__init__()
        self.base = GraphADFCWorm(base_variant, n_nodes, d, fdim, sensor_nodes, motor_nodes, degree)
        self.order_bank = PairwiseOrderBank(fdim, d)
        self.key_bank = KeyReadBank(fdim, d)
        self.order_head = nn.Linear(d, 2)
        self.key_head = nn.Linear(d, 2)
        self.router = nn.Sequential(
            nn.LayerNorm(3 * fdim),
            nn.Linear(3 * fdim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 3),
        )
        self.hard = hard
        self.last_stats: Dict[str, float] = {}

    def route_weights(self, x: torch.Tensor):
        # features include mean, max, and last token; enough to detect query/task structure.
        feat = torch.cat([x.mean(1), x.max(1).values, x[:, -1]], dim=-1)
        logits = self.router(feat)
        soft = torch.softmax(logits, dim=-1)
        if self.hard and self.training:
            idx = soft.argmax(dim=-1)
            hard = F.one_hot(idx, num_classes=3).to(soft.dtype)
            return hard + soft - soft.detach(), logits
        return soft, logits

    def forward(self, x: torch.Tensor):
        graph_logits = self.base(x)
        order_logits = self.order_head(self.order_bank(x))
        key_logits = self.key_head(self.key_bank(x))
        weights, router_logits = self.route_weights(x)
        stack = torch.stack([graph_logits, order_logits, key_logits], dim=1)  # [B,3,2]
        logits = (weights.unsqueeze(-1) * stack).sum(dim=1)
        with torch.no_grad():
            w = weights.detach().mean(0).float().cpu()
            ent = (-(weights.detach().clamp_min(1e-9) * weights.detach().clamp_min(1e-9).log()).sum(-1).mean() / math.log(3)).float().cpu()
            self.last_stats = {
                "w_graph": float(w[0]), "w_order": float(w[1]), "w_key": float(w[2]),
                "router_entropy": float(ent),
                "order_abs": float(self.order_bank.last_abs.cpu()),
                "key_entropy": float(self.key_bank.last_entropy.cpu()),
            }
            self.last_stats.update({"base_" + k: v for k, v in self.base.stats().items()})
        return logits

    def stats(self):
        return dict(self.last_stats)


class AlwaysAddTyped(nn.Module):
    """Reference: graph + order + key logits, no learned router."""
    def __init__(self, base_variant: str, n_nodes: int, d: int, fdim: int, sensor_nodes: int, motor_nodes: int, degree: int):
        super().__init__()
        self.base = GraphADFCWorm(base_variant, n_nodes, d, fdim, sensor_nodes, motor_nodes, degree)
        self.order_bank = PairwiseOrderBank(fdim, d)
        self.key_bank = KeyReadBank(fdim, d)
        self.order_head = nn.Linear(d, 2)
        self.key_head = nn.Linear(d, 2)
        self.scales = nn.Parameter(torch.ones(3))
        self.last_stats = {}
    def forward(self, x):
        gl = self.base(x)
        ol = self.order_head(self.order_bank(x))
        kl = self.key_head(self.key_bank(x))
        w = torch.softmax(self.scales, dim=0)
        logits = w[0] * gl + w[1] * ol + w[2] * kl
        with torch.no_grad():
            self.last_stats = {
                "w_graph": float(w[0].cpu()), "w_order": float(w[1].cpu()), "w_key": float(w[2].cpu()),
                "router_entropy": 1.0,
                "order_abs": float(self.order_bank.last_abs.cpu()),
                "key_entropy": float(self.key_bank.last_entropy.cpu()),
            }
        return logits
    def stats(self): return dict(self.last_stats)


def build_model(name: str, args):
    if name == "graph":
        return GraphADFCWorm("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree)
    if name == "always_typed":
        return AlwaysAddTyped("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree)
    if name == "router":
        return RoutedGraphADFC("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, hard=False)
    if name == "router_hard":
        return RoutedGraphADFC("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, hard=True)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, task_name: str, args, dev, nb: int = 4):
    model.eval(); correct = total = 0; losses = []
    per_task_correct = {n: 0 for n in TASK_NAMES}; per_task_total = {n: 0 for n in TASK_NAMES}
    weight_sums = {n: torch.zeros(3, device=dev) for n in TASK_NAMES}; weight_counts = {n: 0 for n in TASK_NAMES}
    for _ in range(nb):
        if task_name == "mixed":
            x, y, tid = make_mixed(args.eval_batch, args.seq_len, args.fdim, dev)
        else:
            x, y = TASKS[task_name](args.eval_batch, args.seq_len, args.fdim, dev)
            tid = torch.full((args.eval_batch,), TASK_NAMES.index(task_name), device=dev, dtype=torch.long)
        logits = model(x); loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(-1)
        correct += int((pred == y).sum().item()); total += int(y.numel()); losses.append(float(loss.cpu()))
        if hasattr(model, "route_weights"):
            weights, _ = model.route_weights(x)
        else:
            st = model.stats(); weights = torch.tensor([st.get("w_graph", 0), st.get("w_order", 0), st.get("w_key", 0)], device=dev).view(1,3).expand(x.shape[0],3)
        for i, name in enumerate(TASK_NAMES):
            mask = tid == i
            if mask.any():
                per_task_correct[name] += int((pred[mask] == y[mask]).sum().item())
                per_task_total[name] += int(mask.sum().item())
                weight_sums[name] += weights[mask].sum(0)
                weight_counts[name] += int(mask.sum().item())
    out = {"val_loss": sum(losses) / len(losses), "val_acc": correct / max(1,total)}
    for name in TASK_NAMES:
        if per_task_total[name] > 0:
            out[f"acc_{name}"] = per_task_correct[name] / per_task_total[name]
            w = weight_sums[name] / max(1, weight_counts[name])
            out[f"w_graph_{name}"] = float(w[0].cpu()); out[f"w_order_{name}"] = float(w[1].cpu()); out[f"w_key_{name}"] = float(w[2].cpu())
    return out


def train_one(model_name: str, task_name: str, args, dev):
    model = build_model(model_name, args).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows=[]; best={"acc":0.0,"step":0,"loss":999.0}; t0=time.time()
    for step in range(1, args.steps+1):
        model.train()
        if task_name == "mixed":
            x, y, _ = make_mixed(args.batch, args.seq_len, args.fdim, dev)
        else:
            x, y = TASKS[task_name](args.batch, args.seq_len, args.fdim, dev)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        ce = F.cross_entropy(logits.float(), y)
        # light entropy penalty encourages decisive routing, not for non-router models
        if hasattr(model, "route_weights"):
            weights, _ = model.route_weights(x)
            entropy = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(-1).mean() / math.log(3)
            loss = ce + args.router_entropy_penalty * entropy
        else:
            loss = ce
        loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize(); ev = evaluate(model, task_name, args, dev, 4)
            if ev["val_acc"] > best["acc"]: best={"acc":ev["val_acc"],"step":step,"loss":ev["val_loss"]}
            row={"task":task_name,"model":model_name,"step":step,"train_loss":float(loss.detach().cpu()),"train_ce":float(ce.detach().cpu()),"val_loss":ev["val_loss"],"val_acc":ev["val_acc"],"best_val_acc":best["acc"],"best_step":best["step"],"params":nparams(model),"grad_norm":float(gn.detach().cpu()),"sec":time.time()-t0}
            row.update(model.stats()); row.update(ev); rows.append(row)
            extra = ""
            if task_name == "mixed":
                extra = f" route={100*ev.get('acc_route',0):.1f} order={100*ev.get('acc_order',0):.1f} kv={100*ev.get('acc_kv',0):.1f}"
            print(f"[{task_name}] {model_name:12s} {step:04d}/{args.steps} val={100*ev['val_acc']:6.2f}% best={100*best['acc']:6.2f}% wg={row.get('w_graph',0):.2f} wo={row.get('w_order',0):.2f} wk={row.get('w_key',0):.2f} H={row.get('router_entropy',0):.2f}{extra}", flush=True)
    return rows, rows[-1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='results/graph_router')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--batch', type=int, default=192)
    p.add_argument('--eval-batch', type=int, default=256)
    p.add_argument('--seq-len', type=int, default=48)
    p.add_argument('--fdim', type=int, default=16)
    p.add_argument('--nodes', type=int, default=48)
    p.add_argument('--dim', type=int, default=48)
    p.add_argument('--sensor-nodes', type=int, default=16)
    p.add_argument('--motor-nodes', type=int, default=4)
    p.add_argument('--degree', type=int, default=6)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--router-entropy-penalty', type=float, default=0.01)
    p.add_argument('--log-every', type=int, default=30)
    p.add_argument('--tasks', default='mixed')
    p.add_argument('--models', default='graph,always_typed,router,router_hard')
    args=p.parse_args(); dev=require_cuda(); seed_all(args.seed)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True); (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    print('=== Graph-ADFC-Worm edge-type router benchmark ===', flush=True)
    print('gpu', torch.cuda.get_device_name(0), 'torch', torch.__version__, flush=True)
    all_rows=[]; finals=[]
    for task in [x for x in args.tasks.split(',') if x]:
        print('\n--- TASK', task, '---', flush=True)
        for model_name in [x for x in args.models.split(',') if x]:
            rows, fin = train_one(model_name, task, args, dev)
            all_rows += rows; finals.append(fin)
            wcsv(out/'train_rows.csv', all_rows); wcsv(out/'summary_rows.csv', finals)
    winners=[]
    for task in [x for x in args.tasks.split(',') if x]:
        sub=[r for r in finals if r['task']==task]; win=max(sub,key=lambda r:r['best_val_acc'])
        winners.append({'task':task,'winner':win['model'],'winner_acc':win['best_val_acc']})
    wcsv(out/'winners.csv', winners); (out/'summary.json').write_text(json.dumps({'finals':finals,'winners':winners}, indent=2),encoding='utf-8')
    print('\n=== WINNERS ===', flush=True)
    for w in winners: print(f"{w['task']} winner={w['winner']} acc={100*w['winner_acc']:.2f}%", flush=True)
    print('DONE', out.resolve(), flush=True)

if __name__=='__main__': main()
