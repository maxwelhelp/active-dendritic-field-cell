#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph-ADFC-Worm Cost-Aware Router + complex program task.

New task: program
  Sequence contains:
    - mode bit
    - two temporal events A/B
    - four key-value records
    - two candidate query keys q0/q1
  Rule:
    cond = (A_before_B XOR mode)
    selected_key = q0 if cond else q1
    label = value[selected_key]

This requires order relation + conditional routing + key-value recall.

Cost-aware router:
  channels = graph, order, key
  loss = CE + entropy_penalty * H(router) + cost_lambda * E[cost(channel)]

Costs default:
  graph=0.05, key=0.35, order=0.55
"""
from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_adfc_worm import TASKS, GraphADFCWorm, require_cuda, seed_all, nparams, wcsv
from graph_adfc_worm_typed_bank import PairwiseOrderBank, KeyReadBank

TASK_NAMES = ["route", "order", "kv", "program"]


@torch.no_grad()
def make_program(B: int, T: int, Fdim: int, dev):
    x = 0.02 * torch.randn(B, T, Fdim, device=dev)
    rows = torch.arange(B, device=dev)

    # mode controls whether order condition is flipped.
    mode = torch.randint(0, 2, (B,), device=dev)
    x[:, 0, 0] = 1.0 - mode.float()
    x[:, 0, 1] = mode.float()

    # four key-value records, value bit follows key token.
    vals = torch.randint(0, 2, (B, 4), device=dev)
    base = torch.tensor([6, 17, 28, 39], device=dev)
    pos = (base[None, :] + torch.randint(-2, 3, (B, 4), device=dev)).clamp(3, T - 10)
    for k in range(4):
        pk = pos[:, k]
        x[rows, pk, 8 + k] = 1.0
        x[rows, pk + 1, 4 + vals[:, k]] = 1.0

    # events A/B at random order, away from final query zone.
    p1 = torch.randint(5, T - 12, (B,), device=dev)
    p2 = (p1 + torch.randint(2, 14, (B,), device=dev)).clamp(max=T - 8)
    swap = torch.randint(0, 2, (B,), device=dev).bool()
    pa = torch.where(swap, p2, p1)
    pb = torch.where(swap, p1, p2)
    x[rows, pa, 6] = 1.0
    x[rows, pb, 7] = 1.0
    a_before_b = (pa < pb).long()

    # two candidate query keys. The temporal rule selects which query to answer.
    q0 = torch.randint(0, 4, (B,), device=dev)
    q1 = torch.randint(0, 4, (B,), device=dev)
    x[:, -4, 13] = 1.0
    x[rows, T - 3, 8 + q0] = 1.0
    x[:, -2, 14] = 1.0
    x[rows, T - 1, 8 + q1] = 1.0

    cond = (a_before_b + mode).remainder(2)  # 1 => choose q0, 0 => choose q1
    chosen = torch.where(cond.bool(), q0, q1)
    y = vals[rows, chosen].long()
    return x, y


ALL_TASKS = dict(TASKS)
ALL_TASKS["program"] = make_program


def make_mixed(B: int, T: int, Fdim: int, dev, task_names):
    chunks = []; labels = []; tids = []
    base = B // len(task_names)
    sizes = [base] * len(task_names)
    sizes[-1] = B - base * (len(task_names) - 1)
    for tid, (name, n) in enumerate(zip(task_names, sizes)):
        x, y = ALL_TASKS[name](n, T, Fdim, dev)
        chunks.append(x); labels.append(y); tids.append(torch.full((n,), tid, device=dev, dtype=torch.long))
    x = torch.cat(chunks, 0); y = torch.cat(labels, 0); task_id = torch.cat(tids, 0)
    perm = torch.randperm(B, device=dev)
    return x[perm], y[perm], task_id[perm]


class CostAwareRouter(nn.Module):
    def __init__(self, base_variant: str, n_nodes: int, d: int, fdim: int,
                 sensor_nodes: int, motor_nodes: int, degree: int, hard: bool = False):
        super().__init__()
        self.base = GraphADFCWorm(base_variant, n_nodes, d, fdim, sensor_nodes, motor_nodes, degree)
        self.order_bank = PairwiseOrderBank(fdim, d)
        self.key_bank = KeyReadBank(fdim, d)
        self.order_head = nn.Linear(d, 2)
        self.key_head = nn.Linear(d, 2)
        self.router = nn.Sequential(nn.LayerNorm(3 * fdim), nn.Linear(3 * fdim, 80), nn.GELU(), nn.Linear(80, 3))
        self.hard = hard
        self.last_stats = {}

    def route_weights(self, x):
        feat = torch.cat([x.mean(1), x.max(1).values, x[:, -1]], -1)
        logits = self.router(feat)
        soft = torch.softmax(logits, -1)
        if self.hard and self.training:
            idx = soft.argmax(-1)
            hard = F.one_hot(idx, 3).to(soft.dtype)
            return hard + soft - soft.detach(), logits
        return soft, logits

    def forward(self, x):
        graph_logits = self.base(x)
        order_logits = self.order_head(self.order_bank(x))
        key_logits = self.key_head(self.key_bank(x))
        weights, _ = self.route_weights(x)
        logits = (weights.unsqueeze(-1) * torch.stack([graph_logits, order_logits, key_logits], 1)).sum(1)
        with torch.no_grad():
            w = weights.mean(0).detach().float().cpu()
            ent = (-(weights.detach().clamp_min(1e-9) * weights.detach().clamp_min(1e-9).log()).sum(-1).mean() / math.log(3)).float().cpu()
            self.last_stats = {
                "w_graph": float(w[0]), "w_order": float(w[1]), "w_key": float(w[2]),
                "router_entropy": float(ent),
                "order_abs": float(self.order_bank.last_abs.cpu()),
                "key_entropy": float(self.key_bank.last_entropy.cpu()),
            }
        return logits

    def stats(self):
        return dict(self.last_stats)


class AlwaysTyped(nn.Module):
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
        w = torch.softmax(self.scales, 0)
        logits = w[0] * self.base(x) + w[1] * self.order_head(self.order_bank(x)) + w[2] * self.key_head(self.key_bank(x))
        with torch.no_grad():
            self.last_stats = {"w_graph": float(w[0].cpu()), "w_order": float(w[1].cpu()), "w_key": float(w[2].cpu()), "router_entropy": 1.0}
        return logits
    def stats(self): return dict(self.last_stats)


def build_model(name: str, args):
    if name == "graph":
        return GraphADFCWorm("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree)
    if name == "always_typed":
        return AlwaysTyped("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree)
    if name == "cost_router":
        return CostAwareRouter("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, hard=False)
    if name == "cost_router_hard":
        return CostAwareRouter("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, hard=True)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, task_name, args, dev, eval_tasks, nb=4):
    model.eval(); correct = total = 0; losses = []
    per = {n: [0, 0] for n in eval_tasks}; wsum = {n: torch.zeros(3, device=dev) for n in eval_tasks}
    for _ in range(nb):
        if task_name == "mixed":
            x, y, tid = make_mixed(args.eval_batch, args.seq_len, args.fdim, dev, eval_tasks)
        else:
            x, y = ALL_TASKS[task_name](args.eval_batch, args.seq_len, args.fdim, dev)
            tid = torch.zeros(args.eval_batch, device=dev, dtype=torch.long)
        logits = model(x); loss = F.cross_entropy(logits.float(), y); pred = logits.argmax(-1)
        correct += int((pred == y).sum().item()); total += int(y.numel()); losses.append(float(loss.cpu()))
        if hasattr(model, "route_weights"):
            weights, _ = model.route_weights(x)
        else:
            st = model.stats(); weights = torch.tensor([st.get("w_graph",0), st.get("w_order",0), st.get("w_key",0)], device=dev).view(1,3).expand(x.shape[0],3)
        if task_name == "mixed":
            for i, n in enumerate(eval_tasks):
                mask = tid == i
                if mask.any():
                    per[n][0] += int((pred[mask] == y[mask]).sum().item()); per[n][1] += int(mask.sum().item()); wsum[n] += weights[mask].sum(0)
        else:
            n = task_name; per[n][0] += int((pred == y).sum().item()); per[n][1] += int(y.numel()); wsum[n] += weights.sum(0)
    out = {"val_loss": sum(losses) / len(losses), "val_acc": correct / max(1, total)}
    for n in eval_tasks:
        if per[n][1] > 0:
            out[f"acc_{n}"] = per[n][0] / per[n][1]
            w = wsum[n] / per[n][1]
            out[f"w_graph_{n}"] = float(w[0].cpu()); out[f"w_order_{n}"] = float(w[1].cpu()); out[f"w_key_{n}"] = float(w[2].cpu())
    return out


def train_one(model_name, task_name, args, dev, eval_tasks):
    model = build_model(model_name, args).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    costs = torch.tensor([args.cost_graph, args.cost_order, args.cost_key], device=dev)
    rows=[]; best={"acc":0.0,"step":0}; t0=time.time()
    for step in range(1, args.steps+1):
        model.train()
        if task_name == "mixed":
            x, y, _ = make_mixed(args.batch, args.seq_len, args.fdim, dev, eval_tasks)
        else:
            x, y = ALL_TASKS[task_name](args.batch, args.seq_len, args.fdim, dev)
        opt.zero_grad(set_to_none=True); logits = model(x); ce = F.cross_entropy(logits.float(), y); loss = ce
        exp_cost = torch.tensor(0.0, device=dev); entropy = torch.tensor(0.0, device=dev)
        if hasattr(model, "route_weights"):
            weights, _ = model.route_weights(x)
            exp_cost = (weights * costs.view(1,3)).sum(-1).mean()
            entropy = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(-1).mean() / math.log(3)
            loss = ce + args.cost_lambda * exp_cost + args.entropy_penalty * entropy
        loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize(); ev = evaluate(model, task_name, args, dev, eval_tasks, 4)
            if ev["val_acc"] > best["acc"]: best={"acc":ev["val_acc"],"step":step}
            row={"task":task_name,"model":model_name,"step":step,"train_loss":float(loss.detach().cpu()),"train_ce":float(ce.detach().cpu()),"expected_cost":float(exp_cost.detach().cpu()),"entropy_loss":float(entropy.detach().cpu()),"val_loss":ev["val_loss"],"val_acc":ev["val_acc"],"best_val_acc":best["acc"],"best_step":best["step"],"params":nparams(model),"grad_norm":float(gn.detach().cpu()),"sec":time.time()-t0}
            row.update(model.stats()); row.update(ev); rows.append(row)
            extra = " ".join([f"{n}={100*ev.get('acc_'+n,0):.1f}" for n in eval_tasks])
            print(f"[{task_name}] {model_name:16s} {step:04d}/{args.steps} val={100*ev['val_acc']:6.2f}% best={100*best['acc']:6.2f}% cost={row['expected_cost']:.3f} wg={row.get('w_graph',0):.2f} wo={row.get('w_order',0):.2f} wk={row.get('w_key',0):.2f} {extra}", flush=True)
    return rows, rows[-1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='results/graph_cost_router')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--batch', type=int, default=192)
    p.add_argument('--eval-batch', type=int, default=256)
    p.add_argument('--seq-len', type=int, default=56)
    p.add_argument('--fdim', type=int, default=16)
    p.add_argument('--nodes', type=int, default=48)
    p.add_argument('--dim', type=int, default=48)
    p.add_argument('--sensor-nodes', type=int, default=16)
    p.add_argument('--motor-nodes', type=int, default=4)
    p.add_argument('--degree', type=int, default=6)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--cost-graph', type=float, default=0.05)
    p.add_argument('--cost-key', type=float, default=0.35)
    p.add_argument('--cost-order', type=float, default=0.55)
    p.add_argument('--cost-lambda', type=float, default=0.10)
    p.add_argument('--entropy-penalty', type=float, default=0.01)
    p.add_argument('--log-every', type=int, default=40)
    p.add_argument('--tasks', default='mixed')
    p.add_argument('--mixed-tasks', default='route,order,kv,program')
    p.add_argument('--models', default='graph,always_typed,cost_router,cost_router_hard')
    args=p.parse_args(); dev=require_cuda(); seed_all(args.seed)
    eval_tasks=[x for x in args.mixed_tasks.split(',') if x]
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True); (out/'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    print('=== Graph-ADFC-Worm cost-aware router + program task ===', flush=True)
    print('gpu', torch.cuda.get_device_name(0), 'torch', torch.__version__, flush=True)
    all_rows=[]; finals=[]
    for task in [x for x in args.tasks.split(',') if x]:
        print('\n--- TASK', task, '---', flush=True)
        for model_name in [x for x in args.models.split(',') if x]:
            rows, fin = train_one(model_name, task, args, dev, eval_tasks)
            all_rows += rows; finals.append(fin); wcsv(out/'train_rows.csv', all_rows); wcsv(out/'summary_rows.csv', finals)
    winners=[]
    for task in [x for x in args.tasks.split(',') if x]:
        sub=[r for r in finals if r['task']==task]; win=max(sub,key=lambda r:r['best_val_acc'])
        winners.append({'task':task,'winner':win['model'],'winner_acc':win['best_val_acc']})
    wcsv(out/'winners.csv', winners); (out/'summary.json').write_text(json.dumps({'finals':finals,'winners':winners}, indent=2),encoding='utf-8')
    print('\n=== WINNERS ===', flush=True)
    for w in winners: print(f"{w['task']} winner={w['winner']} acc={100*w['winner_acc']:.2f}%", flush=True)
    print('DONE', out.resolve(), flush=True)

if __name__ == '__main__': main()
