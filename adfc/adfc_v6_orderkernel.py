#!/usr/bin/env python3
import argparse,csv,json,math,random,time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# ADFC v3: active dendritic state + keyed dendritic memory readout.
# Not full self-attention: one query reads a linear token memory / shifted-value memory.

def cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: GPU-only')
    return torch.device('cuda')
def seed_all(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
def wcsv(p,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text('')
        return
    ks=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                ks.append(k); seen.add(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,ks); w.writeheader(); w.writerows(rows)

@torch.no_grad()
def mode_select(B,T,V,dev):
    x=torch.randint(32,V,(B,T),device=dev); rows=torch.arange(B,device=dev)
    mode=torch.randint(0,2,(B,),device=dev); a=torch.randint(0,2,(B,),device=dev); b=torch.randint(0,2,(B,),device=dev)
    pa=torch.randint(5,T//2-3,(B,),device=dev); pb=torch.randint(T//2,T-5,(B,),device=dev)
    x[:,0]=1+mode; x[rows,pa]=10; x[rows,pa+1]=20+a; x[rows,pb]=11; x[rows,pb+1]=20+b; x[:,-1]=30
    return x.long(),torch.where(mode==0,a,b).long()
@torch.no_grad()
def order_compare(B,T,V,dev):
    x=torch.randint(32,V,(B,T),device=dev); rows=torch.arange(B,device=dev); mode=torch.randint(0,2,(B,),device=dev)
    p1=torch.randint(4,T-8,(B,),device=dev); p2=(p1+torch.randint(2,16,(B,),device=dev)).clamp(max=T-4)
    sw=torch.randint(0,2,(B,),device=dev).bool(); pa=torch.where(sw,p2,p1); pb=torch.where(sw,p1,p2)
    x[:,0]=3+mode; x[rows,pa]=12; x[rows,pb]=13; x[:,-1]=31
    return x.long(),((pa<pb).long()+mode).remainder(2).long()
@torch.no_grad()
def kv_recall4(B,T,V,dev):
    x=torch.randint(50,V,(B,T),device=dev); rows=torch.arange(B,device=dev)
    vals=torch.randint(0,2,(B,4),device=dev); base=torch.tensor([6,18,30,42],device=dev)
    pos=(base[None,:]+torch.randint(-2,3,(B,4),device=dev)).clamp(3,T-8)
    for k in range(4):
        pk=pos[:,k]; x[rows,pk]=14; x[rows,pk+1]=40+k; x[rows,pk+2]=20+vals[:,k]
    q=torch.randint(0,4,(B,),device=dev); x[:,-2]=15; x[:,-1]=40+q
    return x.long(),vals[rows,q].long()
TASKS={'mode_select':mode_select,'order_compare':order_compare,'kv_recall4':kv_recall4}

class Swi(nn.Module):
    def __init__(self,d):
        super().__init__(); h=4*d; self.n=nn.LayerNorm(d); self.g=nn.Linear(d,h,bias=False); self.u=nn.Linear(d,h,bias=False); self.o=nn.Linear(h,d,bias=False); self.s=nn.Parameter(torch.tensor(.25))
    def forward(self,x):
        z=self.n(x); return x+self.s.tanh()*self.o(F.silu(self.g(z))*self.u(z))
class Flat(nn.Module):
    def __init__(self,V,d,T,depth,readout):
        super().__init__(); self.readout=readout; self.e=nn.Embedding(V,d); self.p=nn.Parameter(torch.randn(T,d)*.02); self.blocks=nn.ModuleList([Swi(d) for _ in range(depth)]); self.n=nn.LayerNorm(d); self.h=nn.Linear(d,2)
    def forward(self,ids):
        x=self.e(ids)+self.p[:ids.shape[1]][None]
        for b in self.blocks: x=b(x)
        x=self.n(x); z=x.mean(1) if self.readout=='mean' else x[:,-1]
        return self.h(z)

class DendState(nn.Module):
    def __init__(self,d,branches,rank):
        super().__init__(); self.b=branches; self.r=rank; kr=branches*rank
        self.n=nn.LayerNorm(d); self.gate=nn.Linear(d,kr,bias=False); self.val=nn.Linear(d,kr,bias=False); self.ctx=nn.Linear(d,kr)
        self.th=nn.Parameter(torch.zeros(branches,rank)); self.decay=nn.Parameter(torch.full((branches,rank),1.0)); self.rec=nn.Linear(kr,kr,bias=False); self.down=nn.Linear(kr,d,bias=False); self.out=nn.Linear(d,d,bias=False); self.rs=nn.Parameter(torch.tensor(.35)); self.last_gate=None
    def forward(self,x):
        B,T,D=x.shape; z=self.n(x); kr=self.b*self.r; h=torch.zeros(B,self.b,self.r,device=x.device,dtype=x.dtype); outs=[]; rates=[]
        mod=self.ctx(z[:,0]+z.mean(1)).view(B,self.b,self.r); decay=torch.sigmoid(self.decay).unsqueeze(0)
        for t in range(T):
            prev=self.rec(h.reshape(B,kr)).view(B,self.b,self.r)
            p=torch.sigmoid(self.gate(z[:,t]).view(B,self.b,self.r)+.5*mod+.25*prev-self.th.unsqueeze(0))
            cand=torch.tanh(self.val(z[:,t]).view(B,self.b,self.r)); h=decay*h+(1-decay)*(p*cand)
            outs.append(self.down(h.reshape(B,kr))); rates.append(p.mean())
        y=torch.stack(outs,1); self.last_gate=torch.stack(rates).mean().detach()
        return x+self.rs.tanh()*self.out(y), h

class KeyedRead(nn.Module):
    def __init__(self,d,mem_d):
        super().__init__(); self.n=nn.LayerNorm(d); self.k=nn.Linear(d,mem_d,bias=False); self.q=nn.Linear(d,mem_d,bias=False); self.v=nn.Linear(d,mem_d,bias=False); self.out=nn.Linear(mem_d,d); self.last_entropy=None; self.scale=mem_d**-0.5
    def forward(self,x):
        z=self.n(x); B,T,D=z.shape
        keys=self.k(z)
        vals=self.v(torch.cat([z[:,1:],z[:,-1:]],1))  # shifted value: key token can read following value token
        q=self.q(z[:,-1]+z[:,-2])
        logits=torch.einsum('bd,btd->bt',q,keys)*self.scale
        att=F.softmax(logits,dim=1)
        read=torch.einsum('bt,btd->bd',att,vals)
        ent=-(att*(att.clamp_min(1e-9).log())).sum(1).mean()/math.log(T)
        self.last_entropy=ent.detach()
        return torch.tanh(self.out(read))

class ADFC2(nn.Module):
    def __init__(self,V,d,T,depth,b,r):
        super().__init__(); self.e=nn.Embedding(V,d); self.p=nn.Parameter(torch.randn(T,d)*.02); self.blocks=nn.ModuleList([DendState(d,b,r) for _ in range(depth)]); self.n=nn.LayerNorm(d); self.sp=nn.Linear(b*r,d); self.h=nn.Linear(2*d,2)
    def forward(self,ids):
        x=self.e(ids)+self.p[:ids.shape[1]][None]; st=None
        for b in self.blocks: x,st=b(x)
        return self.h(torch.cat([self.n(x[:,-1]),torch.tanh(self.sp(st.reshape(st.shape[0],-1)))],-1))
    def stats(self):
        gs=[b.last_gate for b in self.blocks if b.last_gate is not None]
        return {'plateau':float(torch.stack(gs).mean().cpu())} if gs else {}

class ADFC3(nn.Module):
    def __init__(self,V,d,T,depth,b,r,mem_d):
        super().__init__(); self.e=nn.Embedding(V,d); self.p=nn.Parameter(torch.randn(T,d)*.02); self.blocks=nn.ModuleList([DendState(d,b,r) for _ in range(depth)]); self.n=nn.LayerNorm(d); self.sp=nn.Linear(b*r,d); self.kr=KeyedRead(d,mem_d); self.h=nn.Linear(3*d,2)
    def forward(self,ids):
        x=self.e(ids)+self.p[:ids.shape[1]][None]; st=None
        for b in self.blocks: x,st=b(x)
        z=self.n(x[:,-1]); s=torch.tanh(self.sp(st.reshape(st.shape[0],-1))); m=self.kr(x)
        return self.h(torch.cat([z,s,m],-1))
    def stats(self):
        gs=[b.last_gate for b in self.blocks if b.last_gate is not None]
        out={}
        if gs: out['plateau']=float(torch.stack(gs).mean().cpu())
        if self.kr.last_entropy is not None: out['key_entropy']=float(self.kr.last_entropy.cpu())
        return out



class DirectionalOrderKernel(nn.Module):
    """Learned O(T*K) directional relation operator.

    For each learned pair slot k:
      a_t = detector_A_k(x_t)
      b_t = detector_B_k(x_t)
      order_k = sum_t b_t * prefix_sum_a_before_t - a_t * prefix_sum_b_before_t
    This directly represents A-before-B vs B-before-A without hardcoding token ids.
    """
    def __init__(self,d,pairs=4):
        super().__init__(); self.pairs=pairs
        self.n=nn.LayerNorm(d)
        self.det=nn.Linear(d,2*pairs)
        self.out=nn.Linear(4*pairs,d)
        self.last_abs=None; self.last_gate=None
    def forward(self,x):
        z=self.n(x); B,T,D=z.shape; P=self.pairs
        g=torch.sigmoid(self.det(z)).view(B,T,P,2)
        a=g[...,0]; b=g[...,1]
        # exclusive prefix: before current token
        pa=torch.cumsum(a,dim=1)-a
        pb=torch.cumsum(b,dim=1)-b
        ab=(b*pa).sum(1)
        ba=(a*pb).sum(1)
        denom=(a.sum(1)*b.sum(1)+1e-4)
        order=(ab-ba)/denom.clamp_min(1e-4)
        strength=torch.stack([a.max(1).values,b.max(1).values,a.mean(1),b.mean(1)],-1).reshape(B,4*P)
        feat=torch.cat([order,strength[:,P:]],-1) if False else torch.cat([order,a.max(1).values,b.max(1).values,(a.mean(1)-b.mean(1))],-1)
        self.last_abs=order.detach().abs().mean(); self.last_gate=g.detach().mean()
        return torch.tanh(self.out(feat))

class ADFC6(nn.Module):
    def __init__(self,V,d,T,depth,b,r,mem_d,order_pairs=4):
        super().__init__(); self.e=nn.Embedding(V,d); self.p=nn.Parameter(torch.randn(T,d)*.02)
        self.blocks=nn.ModuleList([DendState(d,b,r) for _ in range(depth)])
        self.n=nn.LayerNorm(d); self.sp=nn.Linear(b*r,d); self.kr=KeyedRead(d,mem_d); self.ok=DirectionalOrderKernel(d,order_pairs); self.h=nn.Linear(4*d,2)
    def forward(self,ids):
        x=self.e(ids)+self.p[:ids.shape[1]][None]; st=None
        for b in self.blocks: x,st=b(x)
        z=self.n(x[:,-1]); s=torch.tanh(self.sp(st.reshape(st.shape[0],-1))); m=self.kr(x); o=self.ok(x)
        return self.h(torch.cat([z,s,m,o],-1))
    def stats(self):
        gs=[b.last_gate for b in self.blocks if b.last_gate is not None]
        out={}
        if gs: out['plateau']=float(torch.stack(gs).mean().cpu())
        if self.kr.last_entropy is not None: out['key_entropy']=float(self.kr.last_entropy.cpu())
        if self.ok.last_abs is not None: out['order_abs']=float(self.ok.last_abs.cpu())
        if self.ok.last_gate is not None: out['order_gate']=float(self.ok.last_gate.cpu())
        return out

def build(name,a):
    if name=='flat_last': return Flat(a.vocab,a.dim,a.seq_len,a.depth,'last')
    if name=='mean_pool': return Flat(a.vocab,a.dim,a.seq_len,a.depth,'mean')
    if name=='adfc2': return ADFC2(a.vocab,a.dim,a.seq_len,a.depth,a.branches,a.rank)
    if name=='adfc3': return ADFC3(a.vocab,a.dim,a.seq_len,a.depth,a.branches,a.rank,a.mem_dim)
    if name=='adfc6': return ADFC6(a.vocab,a.dim,a.seq_len,a.depth,a.branches,a.rank,a.mem_dim,a.order_pairs)
    raise ValueError(name)
@torch.no_grad()
def eval_model(m,task,a,dev,nb=4):
    m.eval(); make=TASKS[task]; corr=tot=0; losses=[]
    for _ in range(nb):
        x,y=make(a.eval_batch,a.seq_len,a.vocab,dev); o=m(x); loss=F.cross_entropy(o.float(),y); corr+=int((o.argmax(-1)==y).sum()); tot+=int(y.numel()); losses.append(float(loss.cpu()))
    return sum(losses)/len(losses),corr/max(1,tot)
def train_one(name,task,a,dev):
    m=build(name,a).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay); make=TASKS[task]
    rows=[]; best=(0,999,0); t0=time.time()
    for step in range(1,a.steps+1):
        m.train(); x,y=make(a.batch,a.seq_len,a.vocab,dev); opt.zero_grad(set_to_none=True); o=m(x); loss=F.cross_entropy(o.float(),y); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step==1 or step%a.log_every==0 or step==a.steps:
            torch.cuda.synchronize(); vl,va=eval_model(m,task,a,dev,4)
            if va>best[0]: best=(va,vl,step)
            row={'task':task,'model':name,'step':step,'train_loss':float(loss.detach().cpu()),'val_loss':vl,'val_acc':va,'best_val_acc':best[0],'best_step':best[2],'params':nparams(m),'grad_norm':float(gn.detach().cpu()),'sec':time.time()-t0}; row.update(m.stats() if hasattr(m,'stats') else {}); rows.append(row)
            print(f"[{task}] {name} {step}/{a.steps} loss={row['train_loss']:.4f} val={100*va:.2f}% best={100*best[0]:.2f}% params={row['params']:,} plateau={row.get('plateau',0):.3f} ent={row.get('key_entropy',0):.3f}",flush=True)
    return rows,rows[-1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--out',default='runs/v3'); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--steps',type=int,default=180); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--eval-batch',type=int,default=512); ap.add_argument('--seq-len',type=int,default=64); ap.add_argument('--vocab',type=int,default=96); ap.add_argument('--dim',type=int,default=64); ap.add_argument('--depth',type=int,default=2); ap.add_argument('--branches',type=int,default=12); ap.add_argument('--rank',type=int,default=12); ap.add_argument('--mem-dim',type=int,default=64); ap.add_argument('--lr',type=float,default=2e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--log-every',type=int,default=30); ap.add_argument('--order-pairs',type=int,default=4); ap.add_argument('--tasks',default='mode_select,order_compare,kv_recall4'); ap.add_argument('--models',default='mean_pool,adfc3,adfc6'); a=ap.parse_args()
    if a.device!='cuda': raise RuntimeError('GPU-only')
    dev=cuda(); seed_all(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    print('=== ADFC v6 directional-order GPU benchmark ==='); print('gpu',torch.cuda.get_device_name(0),'torch',torch.__version__)
    all_rows=[]; finals=[]
    for task in [x for x in a.tasks.split(',') if x]:
        print('\n--- TASK',task,'---')
        for name in [x for x in a.models.split(',') if x]:
            rows,fin=train_one(name,task,a,dev); all_rows+=rows; finals.append(fin); wcsv(out/'train_rows.csv',all_rows); wcsv(out/'summary_rows.csv',finals)
    winners=[]
    for task in [x for x in a.tasks.split(',') if x]:
        sub=[r for r in finals if r['task']==task]; ad=next((r for r in sub if r['model']=='adfc6'),None); base=max([r for r in sub if r['model']!='adfc3'],key=lambda r:r['best_val_acc'],default=None); win=max(sub,key=lambda r:r['best_val_acc'])
        winners.append({'task':task,'winner':win['model'],'winner_acc':win['best_val_acc'],'adfc6_acc':ad['best_val_acc'] if ad else None,'best_non_adfc3':base['model'] if base else None,'non_adfc3_acc':base['best_val_acc'] if base else None,'adfc6_margin':(ad['best_val_acc'] if ad else 0)-(base['best_val_acc'] if base else 0)})
    wcsv(out/'winners.csv',winners); (out/'summary.json').write_text(json.dumps({'finals':finals,'winners':winners},indent=2),encoding='utf-8')
    print('\n=== WINNERS ===')
    for w in winners: print(f"{w['task']} winner={w['winner']} adfc6={100*w['adfc6_acc']:.2f}% non_adfc3={w['best_non_adfc3']}:{100*w['non_adfc3_acc']:.2f}% margin={100*w['adfc6_margin']:+.2f}%")
    print('DONE',out.resolve())
if __name__=='__main__': main()
