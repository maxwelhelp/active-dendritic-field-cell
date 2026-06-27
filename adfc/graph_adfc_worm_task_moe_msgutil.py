#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""012 — Task-MoE Message-Utility GraphADFC.

Goal: make specialization visible, not just remember previous baselines.

Only compares:
  - always_typed: known strong fixed mixture baseline
  - task_moe_msgutil: new model with task-conditioned channels + message-gradient graph utility

The model learns:
  task_probs(x) over route/order/kv/program
  channel mix conditioned on task_probs
  graph subgates conditioned on task_probs

This is closer to MoE/specialization:
  different task types can use different channel logic and different graph masks.
"""
from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_adfc_worm import SharedADFCCell, require_cuda, seed_all, nparams, wcsv
from graph_adfc_worm_cost_router import ALL_TASKS, AlwaysTyped, make_mixed
from graph_adfc_worm_typed_bank import PairwiseOrderBank, KeyReadBank

TASKS4 = ["route", "order", "kv", "program"]
TARGET_CH = torch.tensor([
    [0.75, 0.10, 0.15],  # route
    [0.05, 0.90, 0.05],  # order
    [0.05, 0.15, 0.80],  # kv
    [0.05, 0.45, 0.50],  # program
], dtype=torch.float32)


class TaskCondGraph(nn.Module):
    def __init__(self, n_nodes, d, fdim, sensor_nodes, motor_nodes, degree, n_tasks=4):
        super().__init__(); self.n_nodes=n_nodes; self.d=d; self.fdim=fdim; self.sensor_nodes=sensor_nodes; self.motor_nodes=motor_nodes; self.n_tasks=n_tasks
        self.sensor_embed=nn.Parameter(torch.randn(fdim,d)/math.sqrt(d))
        self.node_bias=nn.Parameter(torch.zeros(n_nodes,d))
        self.cell=SharedADFCCell(d); self.out_norm=nn.LayerNorm(d); self.head=nn.Linear(d*motor_nodes,2)
        mask=torch.zeros(n_nodes,n_nodes); rng=random.Random(123); hidden_start=sensor_nodes; motor_start=n_nodes-motor_nodes
        for i in range(n_nodes):
            choices=[j for j in range(n_nodes) if j!=i]
            for j in rng.sample(choices,min(degree,len(choices))): mask[i,j]=1.0
        for dst in range(hidden_start,n_nodes):
            for src in range(sensor_nodes):
                if rng.random()<0.22: mask[dst,src]=1.0
        for dst in range(motor_start,n_nodes):
            for src in range(hidden_start,motor_start):
                if rng.random()<0.35: mask[dst,src]=1.0
        mask.fill_diagonal_(0.0); self.register_buffer('mask',mask)
        self.weight_logits=nn.Parameter(torch.randn(n_nodes,n_nodes)*0.02)
        self.base_gate_logits=nn.Parameter(torch.full((n_nodes,n_nodes),1.0))
        self.task_gate_logits=nn.Parameter(torch.zeros(n_tasks,n_nodes,n_nodes))
        self.register_buffer('utility_ema',torch.zeros(n_tasks,n_nodes,n_nodes))
        self._saved_msgs=[]
        self._saved_A=None
        self.last_stats={}

    def forward(self,x,task_probs):
        self._saved_msgs=[]
        self._saved_A=None
        B,T,Fdim=x.shape; N=self.n_nodes; D=self.d
        base=torch.sigmoid(self.weight_logits)*torch.sigmoid(self.base_gate_logits)*self.mask
        task_gates=torch.sigmoid(self.task_gate_logits)*self.mask.unsqueeze(0)
        tg=torch.einsum('bk,kij->bij',task_probs,task_gates)
        raw=base.unsqueeze(0)*tg
        A=raw/raw.sum(dim=2,keepdim=True).clamp_min(1e-6)
        if self.training:
            self._saved_A=A
        h=self.node_bias.unsqueeze(0).expand(B,N,D).contiguous(); motor_start=N-self.motor_nodes
        for t in range(T):
            u=torch.zeros(B,N,D,device=x.device,dtype=x.dtype)
            inj=x[:,t,:self.sensor_nodes].unsqueeze(-1)*self.sensor_embed[:self.sensor_nodes].unsqueeze(0)
            u[:,:self.sensor_nodes,:]=inj
            msg=torch.einsum('bij,bjd->bid',A,h)
            if self.training:
                msg.retain_grad(); self._saved_msgs.append(msg)
            h=self.cell(h,u,msg)
        z=self.out_norm(h[:,motor_start:,:]).reshape(B,-1)
        with torch.no_grad():
            mean_tg=task_gates.mean(dim=(1,2)).detach().cpu()
            util=self.utility_ema.mean(dim=(1,2)).detach().cpu()
            self.last_stats={f'graph_gate_{TASKS4[i]}':float(mean_tg[i]) for i in range(self.n_tasks)}
            self.last_stats.update({f'edge_util_{TASKS4[i]}':float(util[i]) for i in range(self.n_tasks)})
        return self.head(z)

    def graph_cost(self,task_probs):
        gates=torch.sigmoid(self.task_gate_logits)*self.mask.unsqueeze(0)
        cost_per_task=gates.sum(dim=(1,2))/self.mask.sum().clamp_min(1.0)
        return (task_probs.mean(0)*cost_per_task).sum()

    @torch.no_grad()
    def structural_step(self, task_probs_mean, beta=0.97, cost=0.01, lr=0.08):
        inst=torch.zeros_like(self.utility_ema)
        if self._saved_A is not None and self._saved_msgs:
            rec=None
            for msg in self._saved_msgs:
                if msg.grad is not None:
                    val=(msg.grad.detach()*msg.detach()).abs().mean(dim=2)  # [B,N receiver utility]
                    rec=val if rec is None else rec+val
            if rec is not None:
                rec=rec/len(self._saved_msgs)
                A=self._saved_A.detach()
                per_edge=A*rec.unsqueeze(2)  # receiver utility distributed to incoming edges
                task_mass=task_probs_mean.detach().clamp_min(1e-6)
                if task_mass.sum()>0:
                    task_mass=task_mass/task_mass.sum()
                else:
                    task_mass=torch.full((self.n_tasks,),1.0/self.n_tasks,device=A.device)
                inst=task_mass.view(self.n_tasks,1,1)*per_edge.mean(dim=0).unsqueeze(0)
        else:
            grad=self.task_gate_logits.grad
            if grad is not None:
                inst=(grad.detach()*torch.sigmoid(self.task_gate_logits)).abs()*self.mask.unsqueeze(0)
        self.utility_ema.mul_(beta).add_((1-beta)*inst)
        score=self.utility_ema-cost*torch.sigmoid(self.task_gate_logits)*self.mask.unsqueeze(0)
        self.task_gate_logits.add_(lr*torch.tanh(score*20.0)*self.mask.unsqueeze(0))
        self._saved_msgs=[]; self._saved_A=None

    def stats(self): return dict(self.last_stats)


class TaskMoEUtility(nn.Module):
    def __init__(self,args):
        super().__init__(); self.args=args
        self.task_router=nn.Sequential(nn.LayerNorm(3*args.fdim),nn.Linear(3*args.fdim,80),nn.GELU(),nn.Linear(80,4))
        self.graph=TaskCondGraph(args.nodes,args.dim,args.fdim,args.sensor_nodes,args.motor_nodes,args.degree,4)
        self.order=PairwiseOrderBank(args.fdim,args.dim); self.key=KeyReadBank(args.fdim,args.dim)
        self.order_head=nn.Linear(args.dim,2); self.key_head=nn.Linear(args.dim,2)
        self.last_stats={}
    def task_probs(self,x):
        feat=torch.cat([x.mean(1),x.max(1).values,x[:,-1]],-1)
        return torch.softmax(self.task_router(feat),-1)
    def channel_weights(self,tp):
        target=TARGET_CH.to(tp.device)
        w=tp@target
        return w/w.sum(-1,keepdim=True).clamp_min(1e-6)
    def forward(self,x):
        tp=self.task_probs(x); cw=self.channel_weights(tp)
        gl=self.graph(x,tp); ol=self.order_head(self.order(x)); kl=self.key_head(self.key(x))
        logits=(cw.unsqueeze(-1)*torch.stack([gl,ol,kl],1)).sum(1)
        with torch.no_grad():
            tm=tp.mean(0).cpu(); wm=cw.mean(0).cpu(); ent=(-(tp.clamp_min(1e-9)*tp.clamp_min(1e-9).log()).sum(-1).mean()/math.log(4)).cpu()
            self.last_stats={'task_p_route':float(tm[0]),'task_p_order':float(tm[1]),'task_p_kv':float(tm[2]),'task_p_program':float(tm[3]),'w_graph':float(wm[0]),'w_order':float(wm[1]),'w_key':float(wm[2]),'task_entropy':float(ent),'order_abs':float(self.order.last_abs.cpu()),'key_entropy':float(self.key.last_entropy.cpu())}
            self.last_stats.update(self.graph.stats())
        return logits
    def aux_losses(self,x,tid,eval_tasks):
        tp=self.task_probs(x); true=torch.zeros(x.shape[0],4,device=x.device)
        for local_i,name in enumerate(eval_tasks):
            if name in TASKS4: true[tid==local_i,TASKS4.index(name)]=1.0
        task_loss=F.mse_loss(tp,true)
        graph_cost=self.graph.graph_cost(tp)
        # specialization: task probabilities should be decisive but not globally collapsed
        entropy=-(tp.clamp_min(1e-9)*tp.clamp_min(1e-9).log()).sum(-1).mean()/math.log(4)
        balance=F.mse_loss(tp.mean(0),torch.full((4,),0.25,device=x.device))
        memory_waste=self.channel_weights(tp)[:,2].mean()*self.key.last_entropy.to(x.device)
        return task_loss, graph_cost, entropy, balance, memory_waste
    @torch.no_grad()
    def structural_step(self):
        self.graph.structural_step(torch.zeros(4,device=self.task_gate_logits_device()))
    def task_gate_logits_device(self): return self.graph.task_gate_logits.device
    def stats(self): return dict(self.last_stats)


def build(name,args):
    if name=='always_typed': return AlwaysTyped('learned_sparse',args.nodes,args.dim,args.fdim,args.sensor_nodes,args.motor_nodes,args.degree)
    if name=='task_moe_msgutil': return TaskMoEUtility(args)
    raise ValueError(name)

@torch.no_grad()
def evaluate(model,task,args,dev,eval_tasks,nb=4):
    model.eval(); correct=total=0; losses=[]; per={n:[0,0] for n in eval_tasks}; tpsum={n:torch.zeros(4,device=dev) for n in eval_tasks}; cwsum={n:torch.zeros(3,device=dev) for n in eval_tasks}
    for _ in range(nb):
        if task=='mixed': x,y,tid=make_mixed(args.eval_batch,args.seq_len,args.fdim,dev,eval_tasks)
        else:
            x,y=ALL_TASKS[task](args.eval_batch,args.seq_len,args.fdim,dev); tid=torch.zeros(args.eval_batch,device=dev,dtype=torch.long)
        logits=model(x); loss=F.cross_entropy(logits.float(),y); pred=logits.argmax(-1)
        correct+=int((pred==y).sum()); total+=int(y.numel()); losses.append(float(loss.cpu()))
        if isinstance(model,TaskMoEUtility): tp=model.task_probs(x); cw=model.channel_weights(tp)
        else:
            st=model.stats(); cw=torch.tensor([st.get('w_graph',0),st.get('w_order',0),st.get('w_key',0)],device=dev).view(1,3).expand(x.shape[0],3); tp=torch.zeros(x.shape[0],4,device=dev)
        for i,n in enumerate(eval_tasks):
            m=tid==i
            if m.any(): per[n][0]+=int((pred[m]==y[m]).sum()); per[n][1]+=int(m.sum()); tpsum[n]+=tp[m].sum(0); cwsum[n]+=cw[m].sum(0)
    out={'val_loss':sum(losses)/len(losses),'val_acc':correct/max(1,total)}
    for n in eval_tasks:
        if per[n][1]>0:
            out[f'acc_{n}']=per[n][0]/per[n][1]; tp=tpsum[n]/per[n][1]; cw=cwsum[n]/per[n][1]
            for j,tname in enumerate(TASKS4): out[f'taskp_{tname}_{n}']=float(tp[j].cpu())
            out[f'w_graph_{n}']=float(cw[0].cpu()); out[f'w_order_{n}']=float(cw[1].cpu()); out[f'w_key_{n}']=float(cw[2].cpu())
    return out

def train_one(name,task,args,dev,eval_tasks):
    model=build(name,args).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    rows=[]; best={'acc':0.0,'step':0}; t0=time.time()
    for step in range(1,args.steps+1):
        model.train(); x,y,tid=make_mixed(args.batch,args.seq_len,args.fdim,dev,eval_tasks) if task=='mixed' else (*ALL_TASKS[task](args.batch,args.seq_len,args.fdim,dev), torch.zeros(args.batch,device=dev,dtype=torch.long))
        opt.zero_grad(set_to_none=True); logits=model(x); ce=F.cross_entropy(logits.float(),y); loss=ce
        aux={'task_loss':0.0,'graph_cost':0.0,'task_entropy':0.0,'balance':0.0,'memory_waste':0.0}
        if isinstance(model,TaskMoEUtility):
            tl,gc,en,ba,mw=model.aux_losses(x,tid,eval_tasks if task=='mixed' else [task]); loss=loss+args.task_lambda*tl+args.graph_lambda*gc+args.entropy_lambda*en+args.balance_lambda*ba+args.memory_waste_lambda*mw; aux={k:float(v.detach().cpu()) for k,v in {'task_loss':tl,'graph_cost':gc,'task_entropy':en,'balance':ba,'memory_waste':mw}.items()}
        loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if isinstance(model,TaskMoEUtility) and step>=args.structural_warmup and step%args.structural_every==0:
            with torch.no_grad():
                tp_mean=model.task_probs(x).mean(0)
            model.graph.structural_step(tp_mean, beta=args.utility_beta, cost=args.utility_cost, lr=args.utility_lr)
        if step==1 or step%args.log_every==0 or step==args.steps:
            torch.cuda.synchronize(); ev=evaluate(model,task,args,dev,eval_tasks,4)
            if ev['val_acc']>best['acc']:
                best={'acc':ev['val_acc'],'step':step}
                try:
                    torch.save({'model':model.state_dict(),'step':step,'val_acc':ev['val_acc']}, Path(args.out)/f'best_{name}.pt')
                except Exception:
                    pass
            row={'task':task,'model':name,'step':step,'train_loss':float(loss.detach().cpu()),'train_ce':float(ce.detach().cpu()),'val_loss':ev['val_loss'],'val_acc':ev['val_acc'],'best_val_acc':best['acc'],'best_step':best['step'],'params':nparams(model),'grad_norm':float(gn.detach().cpu()),'sec':time.time()-t0}; row.update(aux); row.update(model.stats()); row.update(ev); rows.append(row)
            extra=' '.join([f'{n}={100*ev.get("acc_"+n,0):.1f}' for n in eval_tasks])
            print(f'[{task}] {name:16s} {step:04d}/{args.steps} val={100*ev["val_acc"]:6.2f}% best={100*best["acc"]:6.2f}% tg={row.get("task_p_route",0):.2f}/{row.get("task_p_order",0):.2f}/{row.get("task_p_kv",0):.2f}/{row.get("task_p_program",0):.2f} wg={row.get("w_graph",0):.2f} wo={row.get("w_order",0):.2f} wk={row.get("w_key",0):.2f} {extra}',flush=True)
    return rows,rows[-1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/012_task_moe_msgutil'); p.add_argument('--seed',type=int,default=7); p.add_argument('--steps',type=int,default=160); p.add_argument('--batch',type=int,default=192); p.add_argument('--eval-batch',type=int,default=256); p.add_argument('--seq-len',type=int,default=56); p.add_argument('--fdim',type=int,default=16); p.add_argument('--nodes',type=int,default=48); p.add_argument('--dim',type=int,default=48); p.add_argument('--sensor-nodes',type=int,default=16); p.add_argument('--motor-nodes',type=int,default=4); p.add_argument('--degree',type=int,default=6); p.add_argument('--lr',type=float,default=2e-3); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--task-lambda',type=float,default=0.10); p.add_argument('--graph-lambda',type=float,default=0.02); p.add_argument('--entropy-lambda',type=float,default=0.005); p.add_argument('--balance-lambda',type=float,default=0.02); p.add_argument('--memory-waste-lambda',type=float,default=0.01); p.add_argument('--utility-beta',type=float,default=0.97); p.add_argument('--utility-cost',type=float,default=0.01); p.add_argument('--utility-lr',type=float,default=0.05); p.add_argument('--structural-warmup',type=int,default=40); p.add_argument('--structural-every',type=int,default=20); p.add_argument('--log-every',type=int,default=40); p.add_argument('--tasks',default='mixed'); p.add_argument('--mixed-tasks',default='route,order,kv,program'); p.add_argument('--models',default='always_typed,task_moe_msgutil')
    args=p.parse_args(); dev=require_cuda(); seed_all(args.seed); eval_tasks=[x for x in args.mixed_tasks.split(',') if x]; out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8'); print('=== 012 Task-MoE Message-Utility GraphADFC ==='); print('gpu',torch.cuda.get_device_name(0),'torch',torch.__version__)
    all_rows=[]; finals=[]
    for task in [x for x in args.tasks.split(',') if x]:
        print('\n--- TASK',task,'---')
        for name in [x for x in args.models.split(',') if x]:
            rows,fin=train_one(name,task,args,dev,eval_tasks); all_rows+=rows; finals.append(fin); wcsv(out/'train_rows.csv',all_rows); wcsv(out/'summary_rows.csv',finals)
    winners=[]
    for task in [x for x in args.tasks.split(',') if x]:
        sub=[r for r in finals if r['task']==task]; win=max(sub,key=lambda r:r['best_val_acc']); winners.append({'task':task,'winner':win['model'],'winner_acc':win['best_val_acc']})
    wcsv(out/'winners.csv',winners); (out/'summary.json').write_text(json.dumps({'finals':finals,'winners':winners},indent=2),encoding='utf-8'); print('\n=== WINNERS ===')
    for w in winners: print(f"{w['task']} winner={w['winner']} acc={100*w['winner_acc']:.2f}%")
    print('DONE',out.resolve())
if __name__=='__main__': main()
