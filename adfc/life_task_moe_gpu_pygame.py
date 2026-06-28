#!/usr/bin/env python3
from __future__ import annotations

import argparse, math, time
from pathlib import Path
import torch

from life_task_moe_gpu import LifeGPUEnv, write_csv
from graph_adfc_worm import require_cuda, nparams


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/015_life_task_moe_gpu_visual')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--agents',type=int,default=512)
    p.add_argument('--food',type=int,default=128)
    p.add_argument('--width',type=int,default=1000)
    p.add_argument('--height',type=int,default=700)
    p.add_argument('--seq-len',type=int,default=40)
    p.add_argument('--nodes',type=int,default=96)
    p.add_argument('--dim',type=int,default=128)
    p.add_argument('--motor-nodes',type=int,default=8)
    p.add_argument('--degree',type=int,default=12)
    p.add_argument('--initial-energy',type=float,default=70)
    p.add_argument('--hunger-energy',type=float,default=28)
    p.add_argument('--reproduce-energy',type=float,default=88)
    p.add_argument('--child-energy',type=float,default=25)
    p.add_argument('--food-energy',type=float,default=18)
    p.add_argument('--food-spawn',type=float,default=0.030)
    p.add_argument('--odor-radius',type=float,default=145)
    p.add_argument('--sense-radius',type=float,default=180)
    p.add_argument('--turn-rate',type=float,default=0.35)
    p.add_argument('--max-speed',type=float,default=5.0)
    p.add_argument('--segments',type=int,default=7)
    p.add_argument('--segment-len',type=float,default=4.0)
    p.add_argument('--segment-phase-lag',type=float,default=0.85)
    p.add_argument('--segment-inertia',type=float,default=0.38)
    p.add_argument('--lateral-slip',type=float,default=0.18)
    p.add_argument('--transfer-energy',type=float,default=8.0)
    p.add_argument('--metabolism',type=float,default=0.035)
    p.add_argument('--move-cost',type=float,default=0.010)
    p.add_argument('--shape-cost',type=float,default=0.012)
    p.add_argument('--wiggle-cost',type=float,default=0.006)
    p.add_argument('--stretch-cost',type=float,default=0.020)
    p.add_argument('--neural-cost',type=float,default=0.018)
    p.add_argument('--max-age',type=float,default=4000)
    p.add_argument('--lr',type=float,default=2e-3)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--cost-lambda',type=float,default=0.002)
    p.add_argument('--connection-lambda',type=float,default=0.001)
    p.add_argument('--task-lambda',type=float,default=0.05)
    p.add_argument('--contrast-tau',type=float,default=0.12)
    p.add_argument('--contrast-lambda',type=float,default=0.03)
    p.add_argument('--balance-lambda',type=float,default=0.02)
    p.add_argument('--task-entropy-min',type=float,default=0.55)
    p.add_argument('--channel-entropy-min',type=float,default=0.75)
    p.add_argument('--shared-loss-weight',type=float,default=0.35)
    p.add_argument('--adapter-loss-weight',type=float,default=1.00)
    p.add_argument('--adapter-init-std',type=float,default=0.02)
    p.add_argument('--adapter-reset-std',type=float,default=0.08)
    p.add_argument('--birth-frac',type=float,default=0.025)
    p.add_argument('--morph-mut-std',type=float,default=0.035)
    p.add_argument('--code-mut-std',type=float,default=0.05)
    p.add_argument('--adapter-inherit-std',type=float,default=0.025)
    p.add_argument('--structural-interval',type=int,default=50)
    p.add_argument('--prune-frac',type=float,default=0.002)
    p.add_argument('--grow-frac',type=float,default=0.002)
    p.add_argument('--ei-flip-prob',type=float,default=0.0005)
    p.add_argument('--agent-cpg-scale',type=float,default=0.15)
    p.add_argument('--hunger-loss-weight',type=float,default=1.5)
    p.add_argument('--risk-loss-weight',type=float,default=2.0)
    p.add_argument('--stall-lambda',type=float,default=0.20)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--fps',type=int,default=45)
    args=p.parse_args()

    import pygame
    dev=require_cuda()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    env=LifeGPUEnv(args,dev)
    print('=== 015 REAL GPU TaskMoE/ADFC Life + pygame ===',flush=True)
    print('device',dev,torch.cuda.get_device_name(0),'params',nparams(env.policy),flush=True)
    print('renderer only: LifeGPUEnv imported; policy lives inside env',flush=True)

    pygame.init()
    screen=pygame.display.set_mode((args.width,args.height))
    pygame.display.set_caption('015 REAL GPU TaskMoE/ADFC Life')
    font=pygame.font.SysFont('monospace',15)
    clock=pygame.time.Clock()
    rows=[]; t0=time.time(); running=True; paused=False
    while running and env.step_i < args.steps:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
            elif ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE: running=False
                elif ev.key==pygame.K_SPACE: paused=not paused
        if not paused:
            st=env.step()
            if env.step_i % args.log_every == 0 or env.step_i == 1:
                st=dict(st); st['sec']=time.time()-t0; rows.append(st); write_csv(out/'train.csv',rows)
                print('step={step} loss={loss:.4f} E={energy_mean:.2f} food={food_alive} w=({w_graph:.2f},{w_order:.2f},{w_key:.2f}) p=({p_forage:.2f},{p_avoid:.2f},{p_pair:.2f},{p_rest:.2f}) shape=({shape_speed:.2f},{shape_tool:.2f},{shape_guard:.2f},{shape_sense:.2f})'.format(**st),flush=True)
        screen.fill((8,8,14))
        with torch.no_grad():
            fx=env.food_x.detach().cpu().numpy(); fy=env.food_y.detach().cpu().numpy(); fa=env.food_alive.detach().cpu().numpy()
            ax=env.x.detach().cpu().numpy(); ay=env.y.detach().cpu().numpy(); aa=env.angle.detach().cpu().numpy(); en=env.energy.detach().cpu().numpy(); sx=env.sex.detach().cpu().numpy(); sgx=env.seg_x.detach().cpu().numpy(); sgy=env.seg_y.detach().cpu().numpy()
        for x,y,ok in zip(fx,fy,fa):
            if ok: pygame.draw.circle(screen,(70,230,90),(int(x),int(y)),3)
        for x,y,ang,e,sex, xs, ys in zip(ax,ay,aa,en,sx,sgx,sgy):
            base=(80,150,255) if sex < 0.5 else (255,100,175)
            h=max(0.0,1.0-min(1.0,e/max(1.0,args.hunger_energy)))
            col=(int(base[0]*(1-h)+255*h),int(base[1]*(1-h)+60*h),int(base[2]*(1-h)+60*h))
            # 3x smaller: real segment body from env.seg_x/env.seg_y, radius 1-2 px
            for k,(px,py) in enumerate(zip(xs,ys)):
                rr = 2 if k == 0 else 1
                shade = max(0.45, 1.0 - 0.07*k)
                cc=(int(col[0]*shade),int(col[1]*shade),int(col[2]*shade))
                pygame.draw.circle(screen, cc, (int(px), int(py)), rr)
                if k > 0:
                    # Do not draw across toroidal wrap boundary.
                    ddx = abs(float(px) - float(xs[k-1]))
                    ddy = abs(float(py) - float(ys[k-1]))
                    if ddx < args.segment_len * 4.0 and ddy < args.segment_len * 4.0:
                        pygame.draw.line(screen, cc, (int(xs[k-1]), int(ys[k-1])), (int(px), int(py)), 1)
            nx=x+math.cos(ang)*3
            ny=y+math.sin(ang)*3
            pygame.draw.circle(screen,(245,245,245),(int(nx),int(ny)),1)
        st=env.last or {}
        lines=[
            f"step={env.step_i} loss={st.get('loss',0):.4f} E={st.get('energy_mean',0):.1f} food={st.get('food_alive',0)}",
            f"w graph/order/key={st.get('w_graph',0):.2f}/{st.get('w_order',0):.2f}/{st.get('w_key',0):.2f}",
            f"tasks={st.get('p_forage',0):.2f}/{st.get('p_avoid',0):.2f}/{st.get('p_pair',0):.2f}/{st.get('p_rest',0):.2f}",
            f"shape={st.get('shape_speed',0):.2f}/{st.get('shape_tool',0):.2f}/{st.get('shape_guard',0):.2f}/{st.get('shape_sense',0):.2f}",
            f"contacts={st.get('contact_events',0)} transfer={st.get('transfer_mean',0):.3f}",
            f"hunger={st.get('hunger_mean',0):.2f} sw={st.get('survival_weight_mean',0):.2f} deaths={st.get('death_events',0)}",
            'renderer only: policy lives inside LifeGPUEnv',
            'SPACE pause | ESC quit',
        ]
        yy=5
        for line in lines:
            screen.blit(font.render(line,True,(235,235,235)),(8,yy)); yy+=17
        pygame.display.flip(); clock.tick(args.fps)
    write_csv(out/'train.csv',rows)
    torch.save(env.policy.state_dict(),out/'policy_final.pt')
    pygame.quit()

if __name__=='__main__':
    main()
