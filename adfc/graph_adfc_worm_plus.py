#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph-ADFC-Worm+ typed connection operators.

This extends graph_adfc_worm.py with optional typed operator channels:
  - directional order channel: A_before_B - B_before_A
  - keyed read channel: query-key reads value-like event

Interpretation: plain synaptic graph edges transmit state; typed edges/operators implement
special relation channels. This tests whether "connections" need types, not just weights.
"""
from __future__ import annotations
import argparse, csv, json, math, random, time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_adfc_worm import TASKS, GraphADFCWorm, require_cuda, seed_all, nparams, wcsv


class DirectionalOrderChannel(nn.Module):
    def __init__(self, fdim: int, d: int, pairs: int = 4):
        super().__init__(); self.pairs = pairs
        self.det = nn.Linear(fdim, 2 * pairs)
        self.out = nn.Linear(3 * pairs, d)
        self.last_abs = None; self.last_gate = None
    def forward(self, x):
        B,T,Fdim = x.shape; P = self.pairs
        g = torch.sigmoid(self.det(x)).view(B,T,P,2)
        a, b = g[...,0], g[...,1]
        pa = torch.cumsum(a, dim=1) - a
        pb = torch.cumsum(b, dim=1) - b
        ab = (b * pa).sum(1)
        ba = (a * pb).sum(1)
        denom = (a.sum(1) * b.sum(1)).clamp_min(1e-4)
        order = (ab - ba) / denom
        feat = torch.cat([order, a.max(1).values, b.max(1).values], -1)
        self.last_abs = order.detach().abs().mean(); self.last_gate = g.detach().mean()
        return torch.tanh(self.out(feat))


class KeyedReadChannel(nn.Module):
    def __init__(self, fdim: int, d: int, mem: int = 64):
        super().__init__()
        self.k = nn.Linear(fdim, mem, bias=False)
        self.q = nn.Linear(fdim, mem, bias=False)
        self.v = nn.Linear(fdim, mem, bias=False)
        self.out = nn.Linear(mem, d)
        self.scale = mem ** -0.5
        self.last_entropy = None
    def forward(self, x):
        B,T,Fdim = x.shape
        keys = self.k(x)
        vals = self.v(torch.cat([x[:,1:], x[:,-1:]], 1))
        q = self.q(x[:,-1] + x[:,-2])
        logits = torch.einsum('bd,btd->bt', q, keys) * self.scale
        att = torch.softmax(logits, dim=1)
        read = torch.einsum('bt,btd->bd', att, vals)
        ent = -(att * att.clamp_min(1e-9).log()).sum(1).mean() / math.log(T)
        self.last_entropy = ent.detach()
        return torch.tanh(self.out(read))


class GraphADFCWormPlus(nn.Module):
    def __init__(self, base_variant: str, n_nodes: int, d: int, fdim: int, sensor_nodes: int, motor_nodes: int, degree: int, use_ops: bool):
        super().__init__()
        self.use_ops = use_ops
        self.base = GraphADFCWorm(base_variant, n_nodes, d, fdim, sensor_nodes, motor_nodes, degree)
        if use_ops:
            self.order = DirectionalOrderChannel(fdim, d, pairs=4)
            self.keyed = KeyedReadChannel(fdim, d, mem=d)
            self.ops_head = nn.Linear(2 * d, 2)
            self.ops_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.order = None; self.keyed = None; self.ops_head = None; self.ops_scale = None
        self.last_stats: Dict[str, float] = {}
    def forward(self, x):
        logits = self.base(x)
        stats = self.base.stats()
        if self.use_ops:
            o = self.order(x); k = self.keyed(x)
            logits = logits + self.ops_scale.tanh() * self.ops_head(torch.cat([o,k], -1))
            stats.update({
                'order_abs': float(self.order.last_abs.cpu()),
                'order_gate': float(self.order.last_gate.cpu()),
                'key_entropy': float(self.keyed.last_entropy.cpu()),
                'ops_scale': float(self.ops_scale.detach().tanh().cpu()),
            })
        self.last_stats = stats
        return logits
    def stats(self): return dict(self.last_stats)


def build_variant(name, args):
    use_ops = name.endswith('_ops')
    base = name.replace('_ops', '')
    return GraphADFCWormPlus(base, args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, use_ops)


@torch.no_grad()
def evaluate(model, task_name, args, dev, nb=4):
    model.eval(); make = TASKS[task_name]
    corr=tot=0; losses=[]
    for _ in range(nb):
        x,y = make(args.eval_batch, args.seq_len, args.fdim, dev)
        logits = model(x); loss = F.cross_entropy(logits.float(), y)
        corr += int((logits.argmax(-1)==y).sum().item()); tot += int(y.numel()); losses.append(float(loss.cpu()))
    return sum(losses)/len(losses), corr/max(1,tot)


def train_one(variant, task_name, args, dev):
    model = build_variant(variant, args).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    make = TASKS[task_name]
    rows=[]; best={'acc':0.0,'loss':999.0,'step':0}; t0=time.time()
    for step in range(1, args.steps+1):
        model.train(); x,y = make(args.batch, args.seq_len, args.fdim, dev)
        opt.zero_grad(set_to_none=True); logits=model(x); loss=F.cross_entropy(logits.float(), y)
        loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            torch.cuda.synchronize(); vl,va=evaluate(model, task_name, args, dev, 4)
            if va>best['acc']: best={'acc':va,'loss':vl,'step':step}
            row={'task':task_name,'variant':variant,'step':step,'train_loss':float(loss.detach().cpu()),'val_loss':vl,'val_acc':va,'best_val_acc':best['acc'],'best_step':best['step'],'params':nparams(model),'grad_norm':float(gn.detach().cpu()),'sec':time.time()-t0}
            row.update(model.stats()); rows.append(row)
            print(f"[{task_name}] {variant:18s} {step:04d}/{args.steps} loss={row['train_loss']:.4f} val={100*va:6.2f}% best={100*best['acc']:6.2f}% ent={row.get('key_entropy',0):.3f} ord={row.get('order_abs',0):.3f}", flush=True)
    return rows, rows[-1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='results/graph_adfc_worm_plus')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=100)
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
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--tasks', default='route,order,kv')
    p.add_argument('--variants', default='none,fixed_random,learned_sparse,learned_sparse_ops,chem_gap,chem_gap_ops')
    args=p.parse_args(); dev=require_cuda(); seed_all(args.seed)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    print('=== Graph-ADFC-Worm+ typed operator benchmark ===', flush=True)
    print('gpu', torch.cuda.get_device_name(0), 'torch', torch.__version__, flush=True)
    all_rows=[]; finals=[]
    for task in [x for x in args.tasks.split(',') if x]:
        print('\n--- TASK', task, '---', flush=True)
        for variant in [x for x in args.variants.split(',') if x]:
            rows,fin=train_one(variant, task, args, dev); all_rows+=rows; finals.append(fin)
            wcsv(out/'train_rows.csv', all_rows); wcsv(out/'summary_rows.csv', finals)
    winners=[]
    for task in [x for x in args.tasks.split(',') if x]:
        sub=[r for r in finals if r['task']==task]
        win=max(sub, key=lambda r:r['best_val_acc'])
        none=next((r for r in sub if r['variant']=='none'), None)
        winners.append({'task':task,'winner':win['variant'],'winner_acc':win['best_val_acc'],'none_acc':none['best_val_acc'] if none else None,'margin_vs_none':win['best_val_acc']-(none['best_val_acc'] if none else 0.0)})
    wcsv(out/'winners.csv', winners)
    (out/'summary.json').write_text(json.dumps({'finals':finals,'winners':winners}, indent=2), encoding='utf-8')
    print('\n=== WINNERS ===', flush=True)
    for w in winners: print(f"{w['task']} winner={w['winner']} acc={100*w['winner_acc']:.2f}% margin_vs_none={100*w['margin_vs_none']:+.2f}%", flush=True)
    print('DONE', out.resolve(), flush=True)

if __name__ == '__main__': main()
