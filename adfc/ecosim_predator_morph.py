#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""014b — Predator/Morphology ADFC life simulation.

This is a more aggressive ecology than 014:
  - cheap static food is almost removed;
  - small prey-food flees from agents and gives limited energy;
  - agents can hunt/steal energy from each other;
  - each agent has its own independent neural network;
  - each agent has inherited/mutated morphology: size, speed, bite force,
    bite radius, armor, digestion, fertility;
  - the brain controls movement, bite, mating, rest, sprint, shield, spike extension;
  - lifetime plasticity reinforces neural paths that increase energy;
  - reproduction crosses over both brain and morphology.

Use:
  python adfc/ecosim_predator_morph.py --visual pygame
  python adfc/ecosim_predator_morph.py --visual none
"""
from __future__ import annotations
import argparse, csv, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))
def clamp(x, lo, hi): return max(lo, min(hi, x))
def wrap_delta(dx, size):
    if dx > size/2: dx -= size
    elif dx < -size/2: dx += size
    return dx
def dist(dx,dy): return math.sqrt(dx*dx+dy*dy)


class MicroBrain:
    def __init__(self, sensor_dim=26, nodes=28, basis_dim=8, action_dim=10, rng=None):
        self.rng = rng or np.random.default_rng()
        self.sensor_dim=sensor_dim; self.nodes=nodes; self.basis_dim=basis_dim; self.action_dim=action_dim
        self.W_in=self.rng.normal(0,0.35,(sensor_dim,nodes)).astype(np.float32)
        self.W_rec=self.rng.normal(0,0.24,(nodes,nodes)).astype(np.float32)
        self.gate_logits=self.rng.normal(-0.7,0.8,(nodes,nodes)).astype(np.float32)
        self.basis=self.rng.normal(0,1/math.sqrt(nodes),(nodes,basis_dim)).astype(np.float32)
        self.W_basis=self.rng.normal(0,0.35,(basis_dim,action_dim)).astype(np.float32)
        self.W_out=self.rng.normal(0,0.30,(nodes,action_dim)).astype(np.float32)
        self.bias=np.zeros(nodes,dtype=np.float32); self.out_bias=np.zeros(action_dim,dtype=np.float32)
        self.h=np.zeros(nodes,dtype=np.float32); self.prev_h=np.zeros(nodes,dtype=np.float32)
        self.last_activity=0.0; self.last_alive_frac=0.0; self.generation=0
    def clone_mutated(self, other=None, mutation=0.035, crossover=0.5, rng=None):
        rng=rng or np.random.default_rng(); c=MicroBrain(self.sensor_dim,self.nodes,self.basis_dim,self.action_dim,rng=rng)
        for name in ['W_in','W_rec','gate_logits','basis','W_basis','W_out','bias','out_bias']:
            a=getattr(self,name)
            if other is not None:
                b=getattr(other,name); m=rng.random(a.shape)<crossover; v=np.where(m,a,b).copy()
            else: v=a.copy()
            v += rng.normal(0,mutation,v.shape).astype(np.float32); setattr(c,name,v.astype(np.float32))
        c.generation=max(self.generation, getattr(other,'generation',0) if other is not None else 0)+1
        return c
    def step(self, obs):
        self.prev_h=self.h.copy(); gates=sigmoid(self.gate_logits)
        rec=(self.h @ (self.W_rec*gates))/math.sqrt(self.nodes); inp=obs @ self.W_in
        cand=np.tanh(inp+rec+self.bias); self.h=0.70*self.h+0.30*cand
        z=self.h@self.basis; raw=self.h@self.W_out+z@self.W_basis+self.out_bias
        self.last_activity=float(np.mean(np.abs(self.h))); self.last_alive_frac=float(np.mean(gates>0.35))
        return {
            'turn': math.tanh(float(raw[0])),
            'thrust': float(sigmoid(raw[1])),
            'bite': float(sigmoid(raw[2])),
            'mate': float(sigmoid(raw[3])),
            'rest': float(sigmoid(raw[4])),
            'sprint': float(sigmoid(raw[5])),
            'shield': float(sigmoid(raw[6])),
            'extend': float(sigmoid(raw[7])),
            'charge': float(sigmoid(raw[8])),
            'mouth': float(sigmoid(raw[9])),
        }
    def cost(self, neural_cost, connection_cost):
        return neural_cost*self.last_activity + connection_cost*self.last_alive_frac
    def plasticity(self, reward, lr=0.003, cost=0.001):
        reward=float(clamp(reward,-1,1)); gates=sigmoid(self.gate_logits); hebb=np.outer(self.prev_h,self.h).astype(np.float32)
        util=np.abs(hebb)*gates
        self.W_rec += (lr*reward*hebb).astype(np.float32)
        self.gate_logits += (2*lr*reward*util - cost*gates).astype(np.float32)
        np.clip(self.W_rec,-2,2,out=self.W_rec); np.clip(self.gate_logits,-6,4,out=self.gate_logits)


@dataclass
class Morphology:
    size: float = 1.0
    speed: float = 1.0
    bite_force: float = 1.0
    bite_radius: float = 1.0
    armor: float = 0.4
    digestion: float = 0.7
    fertility: float = 1.0
    sense: float = 1.0
    def clone_mutated(self, other=None, mutation=0.06, rng=None):
        rng=rng or np.random.default_rng(); child=Morphology()
        for k in self.__dataclass_fields__:
            a=getattr(self,k); b=getattr(other,k) if other is not None else a
            v=a if rng.random()<0.5 else b
            v *= math.exp(float(rng.normal(0,mutation)))
            lo,hi={
                'size':(0.55,2.2),'speed':(0.45,2.2),'bite_force':(0.35,2.6),'bite_radius':(0.45,2.5),
                'armor':(0.0,2.2),'digestion':(0.25,1.4),'fertility':(0.45,1.8),'sense':(0.45,2.0)
            }[k]
            setattr(child,k,clamp(v,lo,hi))
        return child
    def maintenance_cost(self):
        return 0.010*self.size + 0.010*self.speed + 0.012*self.bite_force + 0.010*self.bite_radius + 0.010*self.armor + 0.006*self.sense


@dataclass
class PreyFood:
    x: float; y: float; energy: float; ttl: int; vx: float=0.0; vy: float=0.0

@dataclass
class Agent:
    id: int; x: float; y: float; angle: float; sex: int; brain: MicroBrain; morph: Morphology
    energy: float=60.0; age:int=0; max_age:int=2600; alive:bool=True; cooldown:int=0
    vx:float=0.0; vy:float=0.0; kills:int=0; children:int=0; last_actions:dict=None


class World:
    def __init__(self,args):
        self.a=args; self.rng=np.random.default_rng(args.seed); self.py=random.Random(args.seed)
        self.W=args.width; self.H=args.height; self.step_i=0; self.next_id=0
        self.agents:List[Agent]=[]; self.prey:List[PreyFood]=[]; self.births=0; self.deaths=0; self.kills=0
        for _ in range(args.prey_init): self.spawn_prey()
        for _ in range(args.agents): self.spawn_agent()
    def spawn_prey(self):
        if len(self.prey)>=self.a.prey_max: return
        self.prey.append(PreyFood(self.py.random()*self.W,self.py.random()*self.H,self.py.uniform(5,self.a.prey_energy),self.py.randint(self.a.prey_ttl//2,self.a.prey_ttl)))
    def random_morph(self):
        m=Morphology();
        for k in m.__dataclass_fields__:
            setattr(m,k,clamp(getattr(m,k)*math.exp(self.py.gauss(0,0.25)),0.25,3.0))
        return m
    def spawn_agent(self, brain=None, morph=None, x=None, y=None, energy=None):
        a=Agent(self.next_id, self.py.random()*self.W if x is None else x%self.W, self.py.random()*self.H if y is None else y%self.H,
                self.py.random()*math.tau, self.py.randint(0,1), brain or MicroBrain(rng=self.rng), morph or self.random_morph(),
                self.a.initial_energy if energy is None else energy, max_age=self.py.randint(int(self.a.max_age*0.75),int(self.a.max_age*1.25)))
        self.next_id+=1; self.agents.append(a); return a
    def nearest_prey(self,a):
        best=None; bd=1e18
        for p in self.prey:
            dx=wrap_delta(p.x-a.x,self.W); dy=wrap_delta(p.y-a.y,self.H); d=dx*dx+dy*dy
            if d<bd: bd=d; best=(p,dx,dy,math.sqrt(d))
        return best
    def nearest_agent(self,a, mate=False):
        best=None; bd=1e18
        for b in self.agents:
            if b is a or not b.alive: continue
            if mate and b.sex==a.sex: continue
            dx=wrap_delta(b.x-a.x,self.W); dy=wrap_delta(b.y-a.y,self.H); d=dx*dx+dy*dy
            if d<bd: bd=d; best=(b,dx,dy,math.sqrt(d))
        return best
    def sense(self,a):
        obs=np.zeros(26,dtype=np.float32); sr=self.a.sense_radius*a.morph.sense
        obs[0]=a.energy/max(1,self.a.reproduce_energy); obs[1]=1-min(1,a.energy/max(1,self.a.hunger_energy)); obs[2]=a.age/max(1,a.max_age)
        obs[3]=1 if a.sex else -1; obs[4]=math.cos(a.angle); obs[5]=math.sin(a.angle)
        obs[6]=a.morph.size; obs[7]=a.morph.speed; obs[8]=a.morph.bite_force; obs[9]=a.morph.bite_radius; obs[10]=a.morph.armor; obs[11]=a.morph.sense
        npy=self.nearest_prey(a)
        if npy:
            _,dx,dy,d=npy; obs[12]=dx/sr; obs[13]=dy/sr; obs[14]=max(0,1-d/sr)
        enemy=self.nearest_agent(a,False)
        if enemy:
            b,dx,dy,d=enemy; obs[15]=dx/sr; obs[16]=dy/sr; obs[17]=max(0,1-d/sr); obs[18]=b.energy/max(1,self.a.reproduce_energy); obs[19]=b.morph.size; obs[20]=b.morph.armor
        mate=self.nearest_agent(a,True)
        if mate:
            b,dx,dy,d=mate; obs[21]=dx/sr; obs[22]=dy/sr; obs[23]=max(0,1-d/sr); obs[24]=b.energy/max(1,self.a.reproduce_energy)
        obs[25]=self.py.uniform(-1,1)
        return np.clip(obs,-3,3)
    def update_prey(self):
        for p in list(self.prey):
            # flee nearest agent
            near=None; bd=1e18
            for a in self.agents:
                dx=wrap_delta(p.x-a.x,self.W); dy=wrap_delta(p.y-a.y,self.H); d=dx*dx+dy*dy
                if d<bd: bd=d; near=(dx,dy,math.sqrt(d))
            if near and near[2]<self.a.prey_sense:
                dx,dy,d=near; p.vx += -dx/(d+1e-6)*self.a.prey_speed; p.vy += -dy/(d+1e-6)*self.a.prey_speed
            p.vx=0.90*p.vx+self.py.uniform(-0.15,0.15); p.vy=0.90*p.vy+self.py.uniform(-0.15,0.15)
            p.x=(p.x+p.vx)%self.W; p.y=(p.y+p.vy)%self.H; p.ttl-=1
            if p.ttl<=0 or p.energy<=0: self.prey.remove(p)
        if self.py.random()<self.a.prey_spawn: self.spawn_prey()
    def prey_eat(self,a,actions):
        if actions['mouth']<0.45: return 0
        gain=0; reach=5*a.morph.size+4*a.morph.bite_radius*(0.6+actions['extend'])
        for p in list(self.prey):
            dx=wrap_delta(p.x-a.x,self.W); dy=wrap_delta(p.y-a.y,self.H); d=dist(dx,dy)
            if d<reach:
                take=min(p.energy,self.a.prey_bite*a.morph.digestion*actions['mouth']); p.energy-=take; gain+=take
                if p.energy<=0.1: self.prey.remove(p)
                break
        a.energy += gain*a.morph.digestion; return gain
    def attack(self,a,actions):
        if actions['bite']<0.55 or a.energy<self.a.attack_min_energy: return 0
        ne=self.nearest_agent(a,False)
        if not ne: return 0
        b,dx,dy,d=ne
        reach=self.a.base_attack_radius*a.morph.bite_radius*(0.6+actions['extend']) + 2*a.morph.size
        if d>reach: return 0
        force=self.a.base_attack_force*a.morph.bite_force*actions['bite']*(0.7+actions['charge'])
        defense=1.0+self.a.armor_scale*b.morph.armor*(0.4+(b.last_actions or {}).get('shield',0.0))
        damage=force/defense
        a.energy -= self.a.attack_cost*(0.5+a.morph.bite_force+actions['charge'])
        steal=min(b.energy, damage*self.a.steal_per_damage)
        b.energy-=steal; a.energy+=steal*a.morph.digestion*self.a.cannibal_eff
        if b.energy<=0:
            b.alive=False; a.kills+=1; self.kills+=1
        return steal
    def mate(self,a,actions):
        if actions['mate']<0.52 or a.cooldown>0 or a.energy<self.a.reproduce_energy*a.morph.fertility: return
        nm=self.nearest_agent(a,True)
        if not nm: return
        b,dx,dy,d=nm
        if d>self.a.mate_radius or b.cooldown>0 or b.energy<self.a.reproduce_energy*b.morph.fertility: return
        if not b.last_actions or b.last_actions.get('mate',0)<0.35: return
        child_energy=self.a.child_energy
        brain=a.brain.clone_mutated(b.brain,self.a.mutation,rng=self.rng); morph=a.morph.clone_mutated(b.morph,self.a.morph_mutation,self.rng)
        a.energy-=child_energy*0.55; b.energy-=child_energy*0.55
        self.spawn_agent(brain,morph,a.x+self.py.uniform(-8,8),a.y+self.py.uniform(-8,8),child_energy); a.cooldown=b.cooldown=self.a.reproduce_cooldown
        a.children+=1; b.children+=1; self.births+=1
    def update_agent(self,a):
        before=a.energy; obs=self.sense(a); act=a.brain.step(obs); a.last_actions=act
        rest=act['rest']; sprint=act['sprint']; shield=act['shield']; extend=act['extend']; charge=act['charge']
        active=1-0.50*rest
        speed=self.a.max_speed*a.morph.speed*(0.65+act['thrust'])*(1+0.9*sprint)*active/(a.morph.size**0.35)
        a.angle += act['turn']*self.a.turn_rate*active
        a.vx=0.80*a.vx+math.cos(a.angle)*speed; a.vy=0.80*a.vy+math.sin(a.angle)*speed
        a.x=(a.x+a.vx)%self.W; a.y=(a.y+a.vy)%self.H
        prey_gain=self.prey_eat(a,act); steal=self.attack(a,act); self.mate(a,act)
        tool_cost=self.a.tool_cost*(sprint*a.morph.speed + shield*a.morph.armor + extend*a.morph.bite_radius + charge*a.morph.bite_force)
        move_cost=self.a.move_cost*(abs(a.vx)+abs(a.vy))*a.morph.size
        a.energy -= self.a.metabolism + a.morph.maintenance_cost() + move_cost + tool_cost + a.brain.cost(self.a.neural_cost,self.a.connection_cost)*active
        if a.energy<self.a.hunger_energy: a.energy-=self.a.hunger_penalty
        a.energy=min(a.energy,self.a.energy_cap); a.age+=1
        if a.cooldown>0: a.cooldown-=1
        reward=(a.energy-before)/max(1,self.a.prey_energy)
        a.brain.plasticity(reward,self.a.plastic_lr,self.a.plastic_cost)
        if a.energy<=0 or a.age>a.max_age: a.alive=False
    def step(self):
        self.step_i+=1; self.update_prey(); self.py.shuffle(self.agents)
        for a in list(self.agents):
            if a.alive: self.update_agent(a)
        before=len(self.agents); self.agents=[a for a in self.agents if a.alive]; self.deaths+=before-len(self.agents)
        if len(self.agents)<self.a.min_agents:
            for _ in range(self.a.min_agents-len(self.agents)): self.spawn_agent(energy=self.a.initial_energy*0.8)
    def stats(self):
        A=self.agents or []
        def mean(attr, default=0): return float(np.mean([attr(a) for a in A])) if A else default
        return dict(step=self.step_i,agents=len(A),prey=len(self.prey),births=self.births,deaths=self.deaths,kills=self.kills,
            energy_mean=mean(lambda a:a.energy),energy_max=float(max([a.energy for a in A],default=0)),activity_mean=mean(lambda a:a.brain.last_activity),alive_edges_mean=mean(lambda a:a.brain.last_alive_frac),generation_max=int(max([a.brain.generation for a in A],default=0)),
            size_mean=mean(lambda a:a.morph.size),speed_mean=mean(lambda a:a.morph.speed),force_mean=mean(lambda a:a.morph.bite_force),radius_mean=mean(lambda a:a.morph.bite_radius),armor_mean=mean(lambda a:a.morph.armor),sense_mean=mean(lambda a:a.morph.sense))


def write_csv(path,rows):
    if not rows: return
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,rows[0].keys()); w.writeheader(); w.writerows(rows)

def run_headless(w,args):
    rows=[]
    for _ in range(args.steps):
        w.step()
        if w.step_i%args.log_every==0:
            st=w.stats(); rows.append(st); print(st,flush=True)
    write_csv(args.csv,rows)

def run_pygame(w,args):
    try: import pygame
    except Exception as e:
        print('[warn] pygame unavailable:',e); return run_headless(w,args)
    pygame.init(); screen=pygame.display.set_mode((args.width,args.height)); clock=pygame.time.Clock(); font=pygame.font.SysFont('monospace',15)
    rows=[]; paused=False; running=True
    while running and w.step_i<args.steps:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
            elif ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE: running=False
                if ev.key==pygame.K_SPACE: paused=not paused
                if ev.key==pygame.K_p:
                    for _ in range(15): w.spawn_prey()
        if not paused:
            for _ in range(args.sim_steps_per_frame):
                w.step()
                if w.step_i%args.log_every==0: rows.append(w.stats())
        screen.fill((8,8,14))
        for p in w.prey:
            pygame.draw.circle(screen,(70,220,90),(int(p.x),int(p.y)),3)
        for a in w.agents:
            hunger=1-min(1,a.energy/max(1,args.hunger_energy)); base=(80,150,255) if a.sex==0 else (255,100,175)
            color=(int(base[0]*(1-hunger)+250*hunger),int(base[1]*(1-hunger)+50*hunger),int(base[2]*(1-hunger)+50*hunger))
            sz=int(4+4*a.morph.size); nose=(a.x+math.cos(a.angle)*(sz+5),a.y+math.sin(a.angle)*(sz+5)); left=(a.x+math.cos(a.angle+2.4)*sz,a.y+math.sin(a.angle+2.4)*sz); right=(a.x+math.cos(a.angle-2.4)*sz,a.y+math.sin(a.angle-2.4)*sz)
            pygame.draw.polygon(screen,color,[(int(nose[0]),int(nose[1])),(int(left[0]),int(left[1])),(int(right[0]),int(right[1]))])
            reach=args.base_attack_radius*a.morph.bite_radius*0.7+2*a.morph.size
            pygame.draw.circle(screen,(100,100,100),(int(a.x),int(a.y)),int(reach),1)
            if a.energy>args.reproduce_energy*a.morph.fertility: pygame.draw.circle(screen,(240,240,80),(int(a.x),int(a.y)),sz+2,1)
        st=w.stats(); txt=[f"step={st['step']} agents={st['agents']} prey={st['prey']} births={st['births']} deaths={st['deaths']} kills={st['kills']}",f"E={st['energy_mean']:.1f}/{st['energy_max']:.1f} gen={st['generation_max']} act={st['activity_mean']:.2f} edges={st['alive_edges_mean']:.2f}",f"morph size={st['size_mean']:.2f} speed={st['speed_mean']:.2f} force={st['force_mean']:.2f} radius={st['radius_mean']:.2f} armor={st['armor_mean']:.2f}","SPACE pause | P add prey | ESC quit"]
        y=5
        for t in txt:
            screen.blit(font.render(t,True,(235,235,235)),(8,y)); y+=17
        pygame.display.flip(); clock.tick(args.fps)
    write_csv(args.csv,rows); pygame.quit()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=7); p.add_argument('--width',type=int,default=1000); p.add_argument('--height',type=int,default=700); p.add_argument('--steps',type=int,default=25000); p.add_argument('--agents',type=int,default=55); p.add_argument('--min-agents',type=int,default=12); p.add_argument('--initial-energy',type=float,default=75); p.add_argument('--energy-cap',type=float,default=180); p.add_argument('--hunger-energy',type=float,default=30); p.add_argument('--metabolism',type=float,default=0.024); p.add_argument('--hunger-penalty',type=float,default=0.022); p.add_argument('--move-cost',type=float,default=0.014); p.add_argument('--tool-cost',type=float,default=0.015); p.add_argument('--neural-cost',type=float,default=0.035); p.add_argument('--connection-cost',type=float,default=0.012); p.add_argument('--plastic-lr',type=float,default=0.004); p.add_argument('--plastic-cost',type=float,default=0.0012); p.add_argument('--prey-init',type=int,default=45); p.add_argument('--prey-max',type=int,default=80); p.add_argument('--prey-spawn',type=float,default=0.075); p.add_argument('--prey-energy',type=float,default=18); p.add_argument('--prey-ttl',type=int,default=900); p.add_argument('--prey-speed',type=float,default=0.18); p.add_argument('--prey-sense',type=float,default=110); p.add_argument('--prey-bite',type=float,default=5); p.add_argument('--attack-min-energy',type=float,default=12); p.add_argument('--base-attack-radius',type=float,default=8); p.add_argument('--base-attack-force',type=float,default=5.0); p.add_argument('--attack-cost',type=float,default=1.2); p.add_argument('--steal-per-damage',type=float,default=5.5); p.add_argument('--armor-scale',type=float,default=0.8); p.add_argument('--cannibal-eff',type=float,default=0.90); p.add_argument('--reproduce-energy',type=float,default=78); p.add_argument('--child-energy',type=float,default=28); p.add_argument('--reproduce-cooldown',type=int,default=220); p.add_argument('--mate-radius',type=float,default=18); p.add_argument('--mutation',type=float,default=0.035); p.add_argument('--morph-mutation',type=float,default=0.07); p.add_argument('--max-age',type=int,default=2600); p.add_argument('--sense-radius',type=float,default=190); p.add_argument('--turn-rate',type=float,default=0.30); p.add_argument('--max-speed',type=float,default=1.8); p.add_argument('--visual',choices=['pygame','none'],default='pygame'); p.add_argument('--fps',type=int,default=45); p.add_argument('--sim-steps-per-frame',type=int,default=2); p.add_argument('--log-every',type=int,default=100); p.add_argument('--csv',default='results/014_predator_morph/stats.csv')
    args=p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); w=World(args); run_pygame(w,args) if args.visual=='pygame' else run_headless(w,args)
if __name__=='__main__': main()
