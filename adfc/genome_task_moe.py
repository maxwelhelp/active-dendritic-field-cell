#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""013 — real genome Task-MoE benchmark.

Downloads a small real genome dataset from NCBI RefSeq:
  Escherichia coli K-12 MG1655, GCF_000005845.2_ASM584v2

Builds four binary tasks from FASTA+GFF annotation:
  start  : annotated CDS start vs internal start-like codon
  strand : plus-strand start vs minus-strand start
  frame  : CDS codon-boundary window vs shifted CDS window
  body   : CDS body window vs intergenic window

Compares:
  always_typed_dna : fixed graph/order/key mixture
  task_moe_dna     : task-conditioned MoE over graph/order/key
"""
from __future__ import annotations
import argparse, gzip, json, math, random, time, urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_adfc_worm import SharedADFCCell, require_cuda, seed_all, nparams, wcsv
from graph_adfc_worm_typed_bank import PairwiseOrderBank, KeyReadBank

DNA_TASKS = ["start", "strand", "frame", "body"]
TASK_TARGET = torch.tensor([
    [0.20, 0.45, 0.35],  # start: context/order + key/motif
    [0.20, 0.60, 0.20],  # strand: orientation/order
    [0.20, 0.25, 0.55],  # frame: local periodic/memory-ish
    [0.55, 0.25, 0.20],  # body: graph/content aggregation
], dtype=torch.float32)

BASES = {"A": 0, "C": 1, "G": 2, "T": 3}
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")
START_CODONS = {"ATG", "GTG", "TTG"}

URL_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2"
FNA_URL = URL_BASE + "/GCF_000005845.2_ASM584v2_genomic.fna.gz"
GFF_URL = URL_BASE + "/GCF_000005845.2_ASM584v2_genomic.gff.gz"


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return
    print(f"[download] {url}", flush=True)
    urllib.request.urlretrieve(url, path)


def read_fasta_gz(path: Path) -> str:
    with gzip.open(path, "rt") as f:
        lines = [ln.strip() for ln in f if ln and not ln.startswith(">")]
    return "".join(lines).upper()


def parse_gff_cds(path: Path) -> List[Tuple[int, int, str]]:
    cds = []
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue
            s = int(parts[3]) - 1
            e = int(parts[4])
            strand = parts[6]
            if e - s >= 90 and strand in "+-":
                cds.append((s, e, strand))
    cds.sort()
    return cds


def rc(seq: str) -> str:
    return seq.translate(RC)[::-1].upper()


def build_intergenic(length: int, cds: List[Tuple[int, int, str]]) -> List[Tuple[int, int]]:
    merged = []
    for s, e, _ in cds:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    out = []
    prev = 0
    for s, e in merged:
        if s - prev > 300:
            out.append((prev, s))
        prev = max(prev, e)
    if length - prev > 300:
        out.append((prev, length))
    return out


def make_samples(seq: str, cds: List[Tuple[int, int, str]], win: int, max_per_task: int, seed: int):
    rng = random.Random(seed)
    L = len(seq)
    half = win // 2
    starts_pos, starts_neg = [], []
    internal_start_like = []
    frame_pos, frame_neg = [], []
    body_pos = []

    for s, e, strand in cds:
        if s < half or e + half >= L or e - s < win + 60:
            continue
        start_center = s + 1 if strand == "+" else e - 2
        if half <= start_center < L - half:
            (starts_pos if strand == "+" else starts_neg).append(start_center)
        coding = seq[s:e] if strand == "+" else rc(seq[s:e])
        ncod = len(coding) // 3
        for k in range(5, max(5, ncod - 5)):
            cod = coding[3*k:3*k+3]
            if cod in START_CODONS and rng.random() < 0.15:
                if strand == "+":
                    c = s + 3*k + 1
                else:
                    c = e - 3*k - 2
                if half <= c < L - half and abs(c - start_center) > 60:
                    internal_start_like.append(c)
            if rng.random() < 0.015:
                if strand == "+":
                    c0 = s + 3*k + 1
                    cs = c0 + rng.choice([1, 2])
                else:
                    c0 = e - 3*k - 2
                    cs = c0 - rng.choice([1, 2])
                if half <= c0 < L-half and half <= cs < L-half:
                    frame_pos.append(c0)
                    frame_neg.append(cs)
            if rng.random() < 0.015:
                c = rng.randrange(s + half//2, e - half//2)
                if half <= c < L-half:
                    body_pos.append(c)

    inter = build_intergenic(L, cds)
    body_neg = []
    for a, b in inter:
        for _ in range(max(1, (b-a)//400)):
            c = rng.randrange(a + half, b - half)
            body_neg.append(c)

    rng.shuffle(starts_pos); rng.shuffle(starts_neg); rng.shuffle(internal_start_like)
    rng.shuffle(frame_pos); rng.shuffle(frame_neg); rng.shuffle(body_pos); rng.shuffle(body_neg)

    samples = {
        "start_pos": starts_pos[:max_per_task],
        "start_neg": internal_start_like[:max_per_task],
        "strand_pos": starts_pos[:max_per_task],
        "strand_neg": starts_neg[:max_per_task],
        "frame_pos": frame_pos[:max_per_task],
        "frame_neg": frame_neg[:max_per_task],
        "body_pos": body_pos[:max_per_task],
        "body_neg": body_neg[:max_per_task],
    }
    return samples


class GenomeBatcher:
    def __init__(self, data_dir: Path, win: int, fdim: int, max_per_task: int, seed: int):
        self.data_dir = data_dir
        self.win = win
        self.fdim = fdim
        self.rng = random.Random(seed)
        fna = data_dir / "GCF_000005845.2_ASM584v2_genomic.fna.gz"
        gff = data_dir / "GCF_000005845.2_ASM584v2_genomic.gff.gz"
        download(FNA_URL, fna); download(GFF_URL, gff)
        self.seq = read_fasta_gz(fna)
        self.cds = parse_gff_cds(gff)
        cache = data_dir / f"samples_win{win}_max{max_per_task}_seed{seed}.pt"
        if cache.exists():
            self.samples = torch.load(cache, map_location="cpu", weights_only=False)
        else:
            self.samples = make_samples(self.seq, self.cds, win, max_per_task, seed)
            torch.save(self.samples, cache)
        print("[genome] len", len(self.seq), "cds", len(self.cds), "sample_counts", {k: len(v) for k, v in self.samples.items()}, flush=True)

    def encode_window(self, center: int, task_id: int, dev) -> torch.Tensor:
        half = self.win // 2
        sub = self.seq[center-half:center-half+self.win]
        x = torch.zeros(self.win, self.fdim, device=dev)
        for i, ch in enumerate(sub):
            j = BASES.get(ch)
            if j is not None:
                x[i, j] = 1.0
        # task prompt channels: model knows the question being asked.
        x[:, 12 + task_id] = 1.0
        return x

    def batch_task(self, task: str, B: int, dev):
        tid = DNA_TASKS.index(task)
        pos = self.samples[f"{task}_pos"]
        neg = self.samples[f"{task}_neg"]
        xs, ys = [], []
        for i in range(B):
            label = i % 2
            pool = pos if label == 1 else neg
            c = self.rng.choice(pool)
            xs.append(self.encode_window(c, tid, dev))
            ys.append(label)
        perm = torch.randperm(B, device=dev)
        return torch.stack(xs, 0)[perm], torch.tensor(ys, device=dev, dtype=torch.long)[perm]

    def mixed(self, B: int, dev, tasks: List[str]):
        chunks, labels, tids = [], [], []
        base = B // len(tasks)
        sizes = [base] * len(tasks); sizes[-1] = B - base * (len(tasks)-1)
        for i, (t, n) in enumerate(zip(tasks, sizes)):
            x, y = self.batch_task(t, n, dev)
            chunks.append(x); labels.append(y); tids.append(torch.full((n,), i, device=dev, dtype=torch.long))
        x = torch.cat(chunks, 0); y = torch.cat(labels, 0); tid = torch.cat(tids, 0)
        p = torch.randperm(B, device=dev)
        return x[p], y[p], tid[p]


class GraphChannel(nn.Module):
    def __init__(self, n_nodes, d, fdim, sensor_nodes, motor_nodes, degree):
        super().__init__(); self.n_nodes=n_nodes; self.d=d; self.sensor_nodes=sensor_nodes; self.motor_nodes=motor_nodes
        self.sensor_embed=nn.Parameter(torch.randn(sensor_nodes,d)/math.sqrt(d))
        self.node_bias=nn.Parameter(torch.zeros(n_nodes,d)); self.cell=SharedADFCCell(d,n_nodes)
        hidden_nodes=max(1,n_nodes-sensor_nodes-motor_nodes)
        self.cell_s=SharedADFCCell(d,sensor_nodes)
        self.cell_h=SharedADFCCell(d,hidden_nodes)
        self.cell_m=SharedADFCCell(d,motor_nodes)
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d*motor_nodes,2)
        rng=random.Random(123); mask=torch.zeros(n_nodes,n_nodes)
        for i in range(n_nodes):
            choices=[j for j in range(n_nodes) if j!=i]
            for j in rng.sample(choices,min(degree,len(choices))): mask[i,j]=1
        for dst in range(sensor_nodes,n_nodes):
            for src in range(sensor_nodes):
                if rng.random()<0.25: mask[dst,src]=1
        mask.fill_diagonal_(0); self.register_buffer('mask',mask)
        self.A_logits=nn.Parameter(torch.randn(n_nodes,n_nodes)*0.02)
        self.ei=nn.Parameter(torch.randn(n_nodes)*0.25+0.35)
        self.register_buffer("ei_type", torch.ones(n_nodes))
        self.G_logits=nn.Parameter(torch.randn(n_nodes,n_nodes)*0.01)
        self.gap_scale=nn.Parameter(torch.tensor(0.10))
        with torch.no_grad():
            self.cell.decay[:sensor_nodes].fill_(-1.5)
            self.cell.decay[sensor_nodes:].fill_(0.6)
            self.cell_s.decay.fill_(-1.7)
            self.cell_h.decay.fill_(0.45)
            self.cell_m.decay.fill_(0.15)
            split=max(1,int(0.75*n_nodes))
            self.ei[:split].fill_(1.0)
            self.ei[split:].fill_(-1.0)
            self.ei_type[:split].fill_(1.0)
            self.ei_type[split:].fill_(-1.0)
    def forward(self,x):
        B,T,F=x.shape; N=self.n_nodes; D=self.d
        A_pos=torch.softmax(self.A_logits.masked_fill(self.mask<=0,-1e4),dim=1)
        dale=getattr(self,'ei_type',torch.sign(torch.tanh(self.ei))).to(A_pos.dtype)[None,:]
        A=A_pos*dale
        Graw=0.5*(self.G_logits+self.G_logits.t())
        G=torch.softmax(Graw.masked_fill(self.mask<=0,-1e4),dim=1)
        h=self.node_bias.unsqueeze(0).expand(B,N,D).contiguous(); motor_start=N-self.motor_nodes
        for t in range(T):
            u=torch.zeros(B,N,D,device=x.device,dtype=x.dtype)
            s=x[:,t,:self.sensor_nodes]
            u[:,:self.sensor_nodes,:]=s.unsqueeze(-1)*self.sensor_embed.unsqueeze(0)
            chem=torch.einsum('ij,bjd->bid',A,h)
            gap=torch.einsum('ij,bjd->bid',G,h)
            msg=chem+torch.sigmoid(self.gap_scale)*gap
            hs=self.cell_s(h[:,:self.sensor_nodes],u[:,:self.sensor_nodes],msg[:,:self.sensor_nodes])
            hh=self.cell_h(h[:,self.sensor_nodes:motor_start],u[:,self.sensor_nodes:motor_start],msg[:,self.sensor_nodes:motor_start]) if motor_start>self.sensor_nodes else h[:,self.sensor_nodes:motor_start]
            hm=self.cell_m(h[:,motor_start:],u[:,motor_start:],msg[:,motor_start:])
            h=torch.cat([hs,hh,hm],dim=1)
        with torch.no_grad():
            act=h.detach().norm(dim=-1)
            corr=torch.einsum('bi,bj->ij',act,act)/max(1,act.shape[0])
            prev=getattr(self,'_hebb',None)
            self._hebb=corr if prev is None else 0.97*prev+0.03*corr
        return self.head(self.norm(h[:,motor_start:,:]).reshape(B,-1))

    @torch.no_grad()
    def maybe_switch_ei(self, prob: float = 0.0005):
        if prob <= 0:
            return 0
        flip = torch.rand_like(self.ei_type) < prob
        if flip.any():
            self.ei_type[flip] *= -1.0
            self.ei.data.copy_(self.ei_type.float())
        return int(flip.sum().item())

class AlwaysTypedDNA(nn.Module):
    def __init__(self,args):
        super().__init__(); self.graph=GraphChannel(args.nodes,args.dim,args.fdim,4,args.motor_nodes,args.degree)
        self.order=PairwiseOrderBank(args.fdim,args.dim); self.key=KeyReadBank(args.fdim,args.dim)
        self.oh=nn.Linear(args.dim,2); self.kh=nn.Linear(args.dim,2); self.scales=nn.Parameter(torch.ones(3)); self.last={}
    def forward(self,x):
        w=torch.softmax(self.scales,0); out=w[0]*self.graph(x)+w[1]*self.oh(self.order(x))+w[2]*self.kh(self.key(x))
        self.last={'w_graph':float(w[0].detach().cpu()),'w_order':float(w[1].detach().cpu()),'w_key':float(w[2].detach().cpu())}; return out
    def stats(self): return dict(self.last)


class TaskMoEDNA(nn.Module):
    def __init__(self,args):
        super().__init__(); self.graph=GraphChannel(args.nodes,args.dim,args.fdim,4,args.motor_nodes,args.degree)
        self.order=PairwiseOrderBank(args.fdim,args.dim); self.key=KeyReadBank(args.fdim,args.dim)
        self.oh=nn.Linear(args.dim,2); self.kh=nn.Linear(args.dim,2)
        self.router=nn.Sequential(nn.LayerNorm(3*args.fdim),nn.Linear(3*args.fdim,80),nn.GELU(),nn.Linear(80,4)); self.last={}
    def task_probs(self,x):
        feat=torch.cat([x.mean(1),x.max(1).values,x[:,-1]],-1); return torch.softmax(self.router(feat),-1)
    def chan_weights(self,tp):
        W=TASK_TARGET.to(tp.device); w=tp@W; return w/w.sum(-1,keepdim=True).clamp_min(1e-6)
    def forward(self,x):
        tp=self.task_probs(x); cw=self.chan_weights(tp)
        gl=self.graph(x); ol=self.oh(self.order(x)); kl=self.kh(self.key(x))
        out=(cw.unsqueeze(-1)*torch.stack([gl,ol,kl],1)).sum(1)
        with torch.no_grad():
            tm=tp.mean(0).cpu(); wm=cw.mean(0).cpu()
            self.last={f'task_p_{DNA_TASKS[i]}':float(tm[i]) for i in range(4)}
            self.last.update({'w_graph':float(wm[0]),'w_order':float(wm[1]),'w_key':float(wm[2]),'order_abs':float(self.order.last_abs.cpu()),'key_entropy':float(self.key.last_entropy.cpu())})
        return out
    def aux(self,x,tid):
        tp=self.task_probs(x); true=F.one_hot(tid,4).float(); task_loss=F.mse_loss(tp,true)
        entropy=-(tp.clamp_min(1e-9)*tp.clamp_min(1e-9).log()).sum(-1).mean()/math.log(4)
        return task_loss, entropy
    def stats(self): return dict(self.last)


def build_model(name,args):
    if name=='always_typed_dna': return AlwaysTypedDNA(args)
    if name=='task_moe_dna': return TaskMoEDNA(args)
    raise ValueError(name)

@torch.no_grad()
def evaluate(model,batcher,args,dev,tasks,nb=3):
    model.eval(); corr=tot=0; losses=[]; per={t:[0,0] for t in tasks}; tpsum={t:torch.zeros(4,device=dev) for t in tasks}; cwsum={t:torch.zeros(3,device=dev) for t in tasks}
    for _ in range(nb):
        x,y,tid=batcher.mixed(args.eval_batch,dev,tasks); logits=model(x); loss=F.cross_entropy(logits.float(),y); pred=logits.argmax(-1)
        corr+=int((pred==y).sum()); tot+=int(y.numel()); losses.append(float(loss.cpu()))
        if isinstance(model,TaskMoEDNA): tp=model.task_probs(x); cw=model.chan_weights(tp)
        else:
            st=model.stats(); cw=torch.tensor([st.get('w_graph',0),st.get('w_order',0),st.get('w_key',0)],device=dev).view(1,3).expand(x.shape[0],3); tp=torch.zeros(x.shape[0],4,device=dev)
        for i,t in enumerate(tasks):
            m=tid==i
            if m.any():
                per[t][0]+=int((pred[m]==y[m]).sum()); per[t][1]+=int(m.sum()); tpsum[t]+=tp[m].sum(0); cwsum[t]+=cw[m].sum(0)
    out={'val_loss':sum(losses)/len(losses),'val_acc':corr/max(1,tot)}
    for t in tasks:
        if per[t][1]>0:
            out[f'acc_{t}']=per[t][0]/per[t][1]; tp=tpsum[t]/per[t][1]; cw=cwsum[t]/per[t][1]
            for j,n in enumerate(DNA_TASKS): out[f'taskp_{n}_{t}']=float(tp[j].cpu())
            out[f'w_graph_{t}']=float(cw[0].cpu()); out[f'w_order_{t}']=float(cw[1].cpu()); out[f'w_key_{t}']=float(cw[2].cpu())
    return out


def train_one(name,batcher,args,dev,tasks):
    model=build_model(name,args).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    rows=[]; best={'acc':0.0,'step':0}; t0=time.time()
    for step in range(1,args.steps+1):
        model.train(); x,y,tid=batcher.mixed(args.batch,dev,tasks); opt.zero_grad(set_to_none=True); logits=model(x); ce=F.cross_entropy(logits.float(),y); loss=ce; task_loss=torch.tensor(0.,device=dev); ent=torch.tensor(0.,device=dev)
        if isinstance(model,TaskMoEDNA):
            task_loss,ent=model.aux(x,tid); loss=loss+args.task_lambda*task_loss+args.entropy_lambda*ent
        loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            torch.cuda.synchronize(); ev=evaluate(model,batcher,args,dev,tasks,args.eval_batches)
            if ev['val_acc']>best['acc']:
                best={'acc':ev['val_acc'],'step':step}; torch.save({'model':model.state_dict(),'step':step,'val_acc':ev['val_acc']}, Path(args.out)/f'best_{name}.pt')
            row={'model':name,'step':step,'train_loss':float(loss.detach().cpu()),'train_ce':float(ce.detach().cpu()),'task_loss':float(task_loss.detach().cpu()),'task_entropy':float(ent.detach().cpu()),'val_loss':ev['val_loss'],'val_acc':ev['val_acc'],'best_val_acc':best['acc'],'best_step':best['step'],'params':nparams(model),'grad_norm':float(gn.detach().cpu()),'sec':time.time()-t0}
            row.update(model.stats()); row.update(ev); rows.append(row)
            extra=' '.join([f'{t}={100*ev.get("acc_"+t,0):.1f}' for t in tasks])
            print(f'[{name}] {step:04d}/{args.steps} val={100*ev["val_acc"]:6.2f}% best={100*best["acc"]:6.2f}% wg={row.get("w_graph",0):.2f} wo={row.get("w_order",0):.2f} wk={row.get("w_key",0):.2f} {extra}',flush=True)
    return rows,rows[-1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/013_genome_task_moe'); p.add_argument('--data-dir',default='data/ecoli_k12'); p.add_argument('--seed',type=int,default=7); p.add_argument('--steps',type=int,default=120); p.add_argument('--batch',type=int,default=96); p.add_argument('--eval-batch',type=int,default=128); p.add_argument('--eval-batches',type=int,default=3); p.add_argument('--seq-len',type=int,default=192); p.add_argument('--fdim',type=int,default=16); p.add_argument('--nodes',type=int,default=32); p.add_argument('--dim',type=int,default=32); p.add_argument('--motor-nodes',type=int,default=4); p.add_argument('--degree',type=int,default=5); p.add_argument('--lr',type=float,default=2e-3); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--task-lambda',type=float,default=0.08); p.add_argument('--entropy-lambda',type=float,default=0.002); p.add_argument('--log-every',type=int,default=30); p.add_argument('--models',default='always_typed_dna,task_moe_dna'); p.add_argument('--tasks',default='start,strand,frame,body'); p.add_argument('--max-per-task',type=int,default=12000)
    args=p.parse_args(); dev=require_cuda(); seed_all(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); args.out=str(out); tasks=[t for t in args.tasks.split(',') if t]
    (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    batcher=GenomeBatcher(Path(args.data_dir),args.seq_len,args.fdim,args.max_per_task,args.seed)
    print('=== 013 real genome Task-MoE ==='); print('gpu',torch.cuda.get_device_name(0),'tasks',tasks,flush=True)
    all_rows=[]; finals=[]
    for name in [x for x in args.models.split(',') if x]:
        rows,fin=train_one(name,batcher,args,dev,tasks); all_rows+=rows; finals.append(fin); wcsv(out/'train_rows.csv',all_rows); wcsv(out/'summary_rows.csv',finals)
    win=max(finals,key=lambda r:r['best_val_acc']); wcsv(out/'winners.csv',[{'winner':win['model'],'winner_acc':win['best_val_acc']}]); (out/'summary.json').write_text(json.dumps({'finals':finals,'winner':win['model']},indent=2),encoding='utf-8')
    print('=== WINNER ===',win['model'],100*win['best_val_acc'])

if __name__=='__main__': main()
