#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""009 — compact Metabolic GraphADFC.

Principle tested:
  useful channels/connections should survive; expensive useless ones should die.

This compact version reuses experiment 007 CostAwareRouter and adds:
  - channel survival gates;
  - metabolic channel maintenance cost;
  - sparse graph edge proxy cost;
  - activity/homeostasis proxy;
  - reports survival and cost metrics.

Full structural birth/death can be added after this minimal test proves the signal.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_adfc_worm import GraphADFCWorm, require_cuda, seed_all, nparams, wcsv
from graph_adfc_worm_cost_router import ALL_TASKS, AlwaysTyped, CostAwareRouter, make_mixed

TARGETS = {
    "route":   [0.75, 0.10, 0.15],
    "order":   [0.05, 0.90, 0.05],
    "kv":      [0.05, 0.15, 0.80],
    "program": [0.05, 0.45, 0.50],
}


def target_for_ids(tid, eval_tasks, dev):
    table = torch.tensor([TARGETS.get(n, [1/3,1/3,1/3]) for n in eval_tasks], dtype=torch.float32, device=dev)
    return table[tid]


class MetabolicRouter(CostAwareRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.survival_logits = nn.Parameter(torch.tensor([1.3, 1.3, 1.3]))
        self.last_metabolic = {}

    def survival(self):
        return torch.sigmoid(self.survival_logits)

    def route_weights(self, x):
        weights, logits = super().route_weights(x)
        s = self.survival().view(1, 3)
        weights = weights * s
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights, logits

    def metabolic_terms(self):
        s = self.survival()
        channel_alive = s.mean()
        # Proxy for edge maintenance: average nonzero candidate edge strength if sparse graph has A_logits.
        edge_proxy = torch.tensor(0.0, device=s.device)
        if hasattr(self.base, "A_logits") and self.base.A_logits is not None:
            mask = getattr(self.base, "candidate_mask", torch.ones_like(self.base.A_logits))
            edge_proxy = (torch.sigmoid(self.base.A_logits) * mask).sum() / mask.sum().clamp_min(1.0)
        # Activity proxy: survival-weighted expected channel usage, prevents all channels being free.
        activity_proxy = (s * torch.tensor([0.4, 1.0, 0.8], device=s.device)).mean()
        return channel_alive, edge_proxy, activity_proxy

    def stats(self):
        st = super().stats() if hasattr(super(), "stats") else {}
        s = self.survival().detach().cpu()
        st.update({
            "survive_graph": float(s[0]), "survive_order": float(s[1]), "survive_key": float(s[2]),
        })
        st.update(self.last_metabolic)
        return st


def build(name, args):
    if name == "graph":
        return GraphADFCWorm("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree)
    if name == "always_typed":
        return AlwaysTyped("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree)
    if name == "cost_router":
        return CostAwareRouter("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, hard=False)
    if name == "metabolic":
        return MetabolicRouter("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, hard=False)
    if name == "metabolic_hard":
        return MetabolicRouter("learned_sparse", args.nodes, args.dim, args.fdim, args.sensor_nodes, args.motor_nodes, args.degree, hard=True)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, task, args, dev, eval_tasks, nb=4):
    model.eval(); correct=total=0; losses=[]
    per={n:[0,0] for n in eval_tasks}; wsum={n:torch.zeros(3,device=dev) for n in eval_tasks}
    for _ in range(nb):
        if task == "mixed":
            x,y,tid = make_mixed(args.eval_batch,args.seq_len,args.fdim,dev,eval_tasks)
        else:
            x,y = ALL_TASKS[task](args.eval_batch,args.seq_len,args.fdim,dev)
            tid = torch.zeros(args.eval_batch,device=dev,dtype=torch.long)
        logits=model(x); loss=F.cross_entropy(logits.float(),y); pred=logits.argmax(-1)
        correct += int((pred==y).sum().item()); total += int(y.numel()); losses.append(float(loss.cpu()))
        if hasattr(model,"route_weights"):
            weights,_=model.route_weights(x)
        else:
            st=model.stats() if hasattr(model,"stats") else {}
            weights=torch.tensor([st.get("w_graph",0),st.get("w_order",0),st.get("w_key",0)],device=dev).view(1,3).expand(x.shape[0],3)
        if task == "mixed":
            for i,n in enumerate(eval_tasks):
                m=tid==i
                if m.any():
                    per[n][0]+=int((pred[m]==y[m]).sum().item()); per[n][1]+=int(m.sum().item()); wsum[n]+=weights[m].sum(0)
        else:
            n=task; per[n][0]+=int((pred==y).sum().item()); per[n][1]+=int(y.numel()); wsum[n]+=weights.sum(0)
    out={"val_loss":sum(losses)/len(losses),"val_acc":correct/max(1,total)}
    for n in eval_tasks:
        if per[n][1]>0:
            out[f"acc_{n}"]=per[n][0]/per[n][1]
            w=wsum[n]/per[n][1]
            out[f"w_graph_{n}"]=float(w[0].cpu()); out[f"w_order_{n}"]=float(w[1].cpu()); out[f"w_key_{n}"]=float(w[2].cpu())
    return out


def train_one(name, task, args, dev, eval_tasks):
    model=build(name,args).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    costs=torch.tensor([args.cost_graph,args.cost_order,args.cost_key],device=dev)
    rows=[]; best={"acc":0.0,"step":0}; t0=time.time()
    for step in range(1,args.steps+1):
        model.train()
        if task=="mixed": x,y,tid=make_mixed(args.batch,args.seq_len,args.fdim,dev,eval_tasks)
        else:
            x,y=ALL_TASKS[task](args.batch,args.seq_len,args.fdim,dev); tid=torch.zeros(args.batch,device=dev,dtype=torch.long)
        opt.zero_grad(set_to_none=True); logits=model(x); ce=F.cross_entropy(logits.float(),y); loss=ce
        expected_cost=torch.tensor(0.0,device=dev); target_loss=torch.tensor(0.0,device=dev); metabolic_loss=torch.tensor(0.0,device=dev)
        if hasattr(model,"route_weights"):
            weights,_=model.route_weights(x)
            expected_cost=(weights*costs.view(1,3)).sum(-1).mean()
            target=target_for_ids(tid, eval_tasks if task=="mixed" else [task], dev)
            target_loss=F.mse_loss(weights,target)
            loss=loss+args.cost_lambda*expected_cost+args.target_lambda*target_loss
        if isinstance(model,MetabolicRouter):
            ch,edge,act=model.metabolic_terms()
            homeo=(act-args.homeo_target).pow(2)
            metabolic_loss=args.channel_lambda*ch+args.edge_lambda*edge+args.activity_lambda*act+args.homeo_lambda*homeo
            loss=loss+metabolic_loss
            model.last_metabolic={"channel_alive_cost":float(ch.detach().cpu()),"edge_proxy_cost":float(edge.detach().cpu()),"activity_proxy":float(act.detach().cpu()),"homeo_loss":float(homeo.detach().cpu())}
        loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            torch.cuda.synchronize(); ev=evaluate(model,task,args,dev,eval_tasks,4)
            if ev["val_acc"]>best["acc"]: best={"acc":ev["val_acc"],"step":step}
            row={"task":task,"model":name,"step":step,"train_loss":float(loss.detach().cpu()),"train_ce":float(ce.detach().cpu()),"expected_cost":float(expected_cost.detach().cpu()),"target_loss":float(target_loss.detach().cpu()),"metabolic_loss":float(metabolic_loss.detach().cpu()),"val_loss":ev["val_loss"],"val_acc":ev["val_acc"],"best_val_acc":best["acc"],"best_step":best["step"],"params":nparams(model),"grad_norm":float(gn.detach().cpu()),"sec":time.time()-t0}
            row.update(model.stats() if hasattr(model,"stats") else {}); row.update(ev); rows.append(row)
            extra=" ".join([f"{n}={100*ev.get('acc_'+n,0):.1f}" for n in eval_tasks])
            print(f"[{task}] {name:15s} {step:04d}/{args.steps} val={100*ev['val_acc']:6.2f}% best={100*best['acc']:6.2f}% cost={row['expected_cost']:.3f} met={row['metabolic_loss']:.3f} wg={row.get('w_graph',0):.2f} wo={row.get('w_order',0):.2f} wk={row.get('w_key',0):.2f} {extra}",flush=True)
    return rows,rows[-1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/009_metabolic_mixed")
    p.add_argument("--seed",type=int,default=7); p.add_argument("--steps",type=int,default=240)
    p.add_argument("--batch",type=int,default=192); p.add_argument("--eval-batch",type=int,default=256)
    p.add_argument("--seq-len",type=int,default=56); p.add_argument("--fdim",type=int,default=16)
    p.add_argument("--nodes",type=int,default=48); p.add_argument("--dim",type=int,default=48); p.add_argument("--sensor-nodes",type=int,default=16); p.add_argument("--motor-nodes",type=int,default=4); p.add_argument("--degree",type=int,default=6)
    p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--cost-graph",type=float,default=0.05); p.add_argument("--cost-order",type=float,default=0.55); p.add_argument("--cost-key",type=float,default=0.35)
    p.add_argument("--cost-lambda",type=float,default=0.02); p.add_argument("--target-lambda",type=float,default=0.04)
    p.add_argument("--channel-lambda",type=float,default=0.03); p.add_argument("--edge-lambda",type=float,default=0.04); p.add_argument("--activity-lambda",type=float,default=0.03); p.add_argument("--homeo-lambda",type=float,default=0.01); p.add_argument("--homeo-target",type=float,default=0.08)
    p.add_argument("--log-every",type=int,default=40); p.add_argument("--tasks",default="mixed"); p.add_argument("--mixed-tasks",default="route,order,kv,program"); p.add_argument("--models",default="always_typed,cost_router,metabolic,metabolic_hard")
    args=p.parse_args(); dev=require_cuda(); seed_all(args.seed); eval_tasks=[x for x in args.mixed_tasks.split(',') if x]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    print("=== 009 Metabolic GraphADFC compact ===",flush=True); print("gpu",torch.cuda.get_device_name(0),"torch",torch.__version__,flush=True)
    all_rows=[]; finals=[]
    for task in [x for x in args.tasks.split(',') if x]:
        print("\n--- TASK",task,"---",flush=True)
        for name in [x for x in args.models.split(',') if x]:
            rows,fin=train_one(name,task,args,dev,eval_tasks); all_rows+=rows; finals.append(fin); wcsv(out/"train_rows.csv",all_rows); wcsv(out/"summary_rows.csv",finals)
    winners=[]
    for task in [x for x in args.tasks.split(',') if x]:
        sub=[r for r in finals if r["task"]==task]; win=max(sub,key=lambda r:r["best_val_acc"]); winners.append({"task":task,"winner":win["model"],"winner_acc":win["best_val_acc"]})
    wcsv(out/"winners.csv",winners); (out/"summary.json").write_text(json.dumps({"finals":finals,"winners":winners},indent=2),encoding="utf-8")
    print("\n=== WINNERS ===",flush=True)
    for w in winners: print(f"{w['task']} winner={w['winner']} acc={100*w['winner_acc']:.2f}%",flush=True)
    print("DONE",out.resolve(),flush=True)

if __name__=="__main__": main()
