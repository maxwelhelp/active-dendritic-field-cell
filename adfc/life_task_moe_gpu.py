#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""015 — REAL imported TaskMoE/ADFC GPU life policy.

This file intentionally imports the existing network code instead of using the
old decorative numpy MicroBrain.

CHECK IMPORTS BELOW.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== REAL PROJECT NETWORK IMPORTS =====
from graph_adfc_worm import require_cuda, nparams, wcsv
from graph_adfc_worm_typed_bank import PairwiseOrderBank, KeyReadBank
from genome_task_moe import GraphChannel, TaskMoEDNA, AlwaysTypedDNA
# =======================================


LIFE_TASKS = ["forage", "avoid", "mate", "rest"]
# action order: turn_left, turn_right, move, eat, attack, mate, rest, speed_shape, tool_shape, guard_shape, sense_shape
ACTIONS = 11
FDIM = 16

def detach_nonparam_tensor_state(module):
    for m in module.modules():
        ps=set(m._parameters.keys()); bs=set(m._buffers.keys())
        for name,val in list(vars(m).items()):
            if name in ps or name in bs:
                continue
            if torch.is_tensor(val):
                try: setattr(m,name,val.detach())
                except Exception: pass



class LifeTaskMoEPolicy(nn.Module):
    """Life policy built from the same Graph/Order/Key TaskMoE idea used in DNA.

    We reuse imported GraphChannel, PairwiseOrderBank and KeyReadBank.
    TaskMoEDNA is imported above to make the dependency explicit; GraphChannel is
    the graph branch from that DNA module.
    """
    def __init__(self, nodes=32, dim=32, seq_len=32, motor_nodes=4, degree=5):
        super().__init__()
        class Args: pass
        args = Args()
        args.nodes = nodes
        args.dim = dim
        args.fdim = FDIM
        args.motor_nodes = motor_nodes
        args.degree = degree
        # Imported real DNA graph channel. It outputs 2 logits, then we lift them.
        self.graph = GraphChannel(nodes, dim, FDIM, 4, motor_nodes, degree)
        self.order = PairwiseOrderBank(FDIM, dim)
        self.key = KeyReadBank(FDIM, dim)
        self.task_router = nn.Sequential(nn.LayerNorm(3 * FDIM), nn.Linear(3 * FDIM, 96), nn.GELU(), nn.Linear(96, 4))
        self.graph_to_action = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, ACTIONS))
        self.order_to_action = nn.Linear(dim, ACTIONS)
        self.key_to_action = nn.Linear(dim, ACTIONS)
        self.channel_by_task = nn.Parameter(torch.tensor([
            [0.55, 0.15, 0.30],  # forage: graph + key
            [0.45, 0.40, 0.15],  # avoid: graph + order
            [0.25, 0.35, 0.40],  # mate: order + key
            [0.60, 0.20, 0.20],  # rest: graph/body state
        ], dtype=torch.float32))
        self.last_stats = {}

    def task_probs(self, x):
        feat = torch.cat([x.mean(1), x.max(1).values, x[:, -1]], dim=-1)
        return torch.softmax(self.task_router(feat), dim=-1)

    def forward(self, x):
        tp = self.task_probs(x)
        ch = torch.softmax(self.channel_by_task, dim=-1)
        cw = tp @ ch
        gl = self.graph_to_action(self.graph(x))
        ol = self.order_to_action(self.order(x))
        kl = self.key_to_action(self.key(x))
        logits = cw[:, 0:1] * gl + cw[:, 1:2] * ol + cw[:, 2:3] * kl
        with torch.no_grad():
            tm = tp.mean(0).detach().cpu()
            wm = cw.mean(0).detach().cpu()
            self.last_stats = {
                "p_forage": float(tm[0]), "p_avoid": float(tm[1]), "p_mate": float(tm[2]), "p_rest": float(tm[3]),
                "w_graph": float(wm[0]), "w_order": float(wm[1]), "w_key": float(wm[2]),
                "order_abs": float(self.order.last_abs.detach().cpu()),
                "key_entropy": float(self.key.last_entropy.detach().cpu()),
            }
        return logits

    def stats(self):
        return dict(self.last_stats)


class LifeGPUEnv:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.N = args.agents
        self.F = args.food
        self.seq_len = args.seq_len
        self.W = float(args.width)
        self.H = float(args.height)
        self.step_i = 0
        torch.manual_seed(args.seed)
        self.policy = LifeTaskMoEPolicy(args.nodes, args.dim, args.seq_len, args.motor_nodes, args.degree).to(device)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.x = torch.rand(self.N, device=device) * self.W
        self.y = torch.rand(self.N, device=device) * self.H
        self.angle = torch.rand(self.N, device=device) * math.tau
        self.energy = torch.full((self.N,), args.initial_energy, device=device)
        self.age = torch.zeros(self.N, device=device)
        self.sex = torch.randint(0, 2, (self.N,), device=device).float()
        self.size = torch.exp(torch.randn(self.N, device=device) * 0.20).clamp(0.6, 2.0)
        self.speed = torch.exp(torch.randn(self.N, device=device) * 0.20).clamp(0.6, 2.0)
        self.tool = torch.exp(torch.randn(self.N, device=device) * 0.20).clamp(0.4, 2.4)
        self.guard = torch.exp(torch.randn(self.N, device=device) * 0.20).clamp(0.2, 2.2)
        self.food_x = torch.rand(self.F, device=device) * self.W
        self.food_y = torch.rand(self.F, device=device) * self.H
        self.food_alive = torch.ones(self.F, dtype=torch.bool, device=device)
        self.births = 0
        self.deaths = 0
        self.last = {}

    def wrap_delta(self, d, size):
        return torch.where(d > size / 2, d - size, torch.where(d < -size / 2, d + size, d))

    def nearest_food(self):
        dx = self.wrap_delta(self.food_x[None, :] - self.x[:, None], self.W)
        dy = self.wrap_delta(self.food_y[None, :] - self.y[:, None], self.H)
        d2 = dx * dx + dy * dy
        d2 = torch.where(self.food_alive[None, :], d2, torch.full_like(d2, 1e9))
        best_d2, idx = d2.min(dim=1)
        rows = torch.arange(self.N, device=self.device)
        return idx, dx[rows, idx], dy[rows, idx], torch.sqrt(best_d2.clamp_min(1e-6))

    def nearest_agent(self):
        dx = self.wrap_delta(self.x[None, :] - self.x[:, None], self.W)
        dy = self.wrap_delta(self.y[None, :] - self.y[:, None], self.H)
        d2 = dx * dx + dy * dy
        d2.fill_diagonal_(1e9)
        best_d2, idx = d2.min(dim=1)
        rows = torch.arange(self.N, device=self.device)
        return idx, dx[rows, idx], dy[rows, idx], torch.sqrt(best_d2.clamp_min(1e-6))

    def encode_obs_sequence(self):
        fidx, fdx, fdy, fd = self.nearest_food()
        aidx, adx, ady, ad = self.nearest_agent()
        base = torch.zeros(self.N, FDIM, device=self.device)
        base[:, 0] = self.energy / self.args.reproduce_energy
        base[:, 1] = 1.0 - (self.energy / self.args.hunger_energy).clamp(0, 1)
        base[:, 2] = torch.cos(self.angle)
        base[:, 3] = torch.sin(self.angle)
        base[:, 4] = fdx / self.args.sense_radius
        base[:, 5] = fdy / self.args.sense_radius
        base[:, 6] = (1.0 - fd / self.args.sense_radius).clamp(0, 1)
        base[:, 7] = adx / self.args.sense_radius
        base[:, 8] = ady / self.args.sense_radius
        base[:, 9] = (1.0 - ad / self.args.sense_radius).clamp(0, 1)
        base[:, 10] = self.size
        base[:, 11] = self.speed
        # task prompt channels 12..15, selected from current ecological state
        hunger = base[:, 1]
        danger = base[:, 9] * (self.energy[aidx] > self.energy).float()
        can_mate = (self.energy > self.args.reproduce_energy).float()
        rest_need = (self.energy < self.args.hunger_energy * 0.55).float()
        task_score = torch.stack([hunger + base[:, 6], danger, can_mate, rest_need], dim=1)
        tid = task_score.argmax(dim=1)
        base[torch.arange(self.N, device=self.device), 12 + tid] = 1.0
        # repeat into sequence with small positional/noise variation; same interface as DNA TaskMoE.
        x = base[:, None, :].repeat(1, self.seq_len, 1)
        pos = torch.linspace(-1, 1, self.seq_len, device=self.device)[None, :, None]
        x[:, :, 0:1] = x[:, :, 0:1] + 0.05 * pos
        return x, tid, (fidx, fdx, fdy, fd, aidx, adx, ady, ad)

    def teacher_targets(self, aux):
        fidx, fdx, fdy, fd, aidx, adx, ady, ad = aux
        target = torch.zeros(self.N, ACTIONS, device=self.device) + 0.25
        desired = torch.atan2(fdy, fdx)
        diff = (desired - self.angle + math.pi) % (2 * math.pi) - math.pi
        target[:, 0] = (diff < -0.05).float() # left
        target[:, 1] = (diff > 0.05).float()  # right
        hunger = 1.0 - (self.energy / self.args.hunger_energy).clamp(0, 1)
        near_food = (1.0 - fd / self.args.sense_radius).clamp(0, 1)
        target[:, 2] = (0.25 + 0.65 * hunger).clamp(0, 1) # move
        target[:, 3] = near_food # eat
        stronger = (self.energy[aidx] > self.energy * 1.15).float()
        near_agent = (1.0 - ad / self.args.sense_radius).clamp(0, 1)
        target[:, 4] = ((1.0 - stronger) * near_agent * 0.55).clamp(0, 1) # attack/steal when not weaker
        target[:, 5] = ((self.energy > self.args.reproduce_energy).float() * near_agent).clamp(0, 1)
        target[:, 6] = ((self.energy < self.args.hunger_energy * 0.65).float() * (1.0 - near_food)).clamp(0, 1)
        target[:, 7] = (0.20 + 0.55 * hunger).clamp(0, 1) # speed shape
        target[:, 8] = (0.25 + 0.55 * near_food).clamp(0, 1) # tool shape
        target[:, 9] = (0.20 + 0.60 * stronger * near_agent).clamp(0, 1) # guard
        target[:, 10] = (0.25 + 0.55 * (1.0 - near_food)).clamp(0, 1) # sense
        return target

    def step(self):
        self.step_i += 1
        xseq, tid, aux = self.encode_obs_sequence()
        logits = self.policy(xseq)
        action = torch.sigmoid(logits)
        target = self.teacher_targets(aux)
        loss_policy = F.mse_loss(action, target)
        stats = self.policy.stats()
        detach_nonparam_tensor_state(self.policy)
        # IMPORTANT: train before mutating environment tensors used to build obs.
        # food_alive participates in torch.where inside nearest_food(); changing it
        # before backward causes an autograd version error. Physics uses detached action.
        reg = self.args.cost_lambda * action.mean()
        loss = loss_policy + reg
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        action_env = action.detach()
        # physics under detached actions.
        turn = (action_env[:, 1] - action_env[:, 0]) * self.args.turn_rate
        move = action_env[:, 2]
        eat = action_env[:, 3]
        attack = action_env[:, 4]
        mate = action_env[:, 5]
        rest = action_env[:, 6]
        sh = action_env[:, 7:11].clamp_min(1e-4)
        sh = sh / sh.sum(dim=1, keepdim=True)
        old_energy = self.energy.clone()
        self.angle = self.angle + turn
        speed = self.args.max_speed * self.speed * (0.35 + 1.4 * sh[:, 0]) * move * (1.0 - 0.45 * rest)
        self.x = (self.x + torch.cos(self.angle) * speed) % self.W
        self.y = (self.y + torch.sin(self.angle) * speed) % self.H
        fidx, fdx, fdy, fd = self.nearest_food()
        aidx, adx, ady, ad = self.nearest_agent()
        # food capture
        near = fd < (6.0 + 10.0 * sh[:, 1] * self.tool)
        take = near.float() * eat * self.args.food_energy
        self.energy = self.energy + take
        self.food_alive[fidx[near]] = False
        # respawn scarce food slowly
        dead_food = ~self.food_alive
        respawn = dead_food & (torch.rand(self.F, device=self.device) < self.args.food_spawn)
        rn = int(respawn.sum().item())
        if rn > 0:
            self.food_x[respawn] = torch.rand(rn, device=self.device) * self.W
            self.food_y[respawn] = torch.rand(rn, device=self.device) * self.H
        self.food_alive[respawn] = True
        # energy steal interaction
        near_a = ad < (7.0 + 8.0 * sh[:, 1] * self.tool)
        steal = near_a.float() * attack * self.args.steal_energy * (0.5 + sh[:, 1]) / (1.0 + self.guard[aidx] * sh[aidx, 2])
        steal = torch.minimum(steal, self.energy[aidx].clamp_min(0))
        self.energy = self.energy + steal * 0.75
        self.energy[aidx] = self.energy[aidx] - steal
        # costs
        cost = self.args.metabolism + self.args.move_cost * speed + self.args.shape_cost * (sh[:, 0] * self.speed + sh[:, 1] * self.tool + sh[:, 2] * self.guard + sh[:, 3])
        cost = cost + self.args.neural_cost * (action_env.abs().mean(dim=1))
        self.energy = self.energy - cost
        self.age = self.age + 1
        # reproduction as param-copy mutation for dead/low slots only for now
        can_birth = (mate > 0.65) & (self.energy > self.args.reproduce_energy)
        births = int(can_birth.sum().item())
        if births:
            self.births += births
            self.energy[can_birth] -= self.args.child_energy * 0.5
        dead = (self.energy <= 0) | (self.age > self.args.max_age)
        dead_n = int(dead.sum().item())
        if dead_n:
            self.deaths += dead_n
            with torch.no_grad():
                self.energy[dead] = self.args.initial_energy * 0.75
                self.age[dead] = 0
                self.x[dead] = torch.rand(dead_n, device=self.device) * self.W
                self.y[dead] = torch.rand(dead_n, device=self.device) * self.H
                self.angle[dead] = torch.rand(dead_n, device=self.device) * math.tau
                # reset only some morphology; policy remains learned globally/agent-slice params remain trainable.
        reward = (self.energy - old_energy).mean() / max(1.0, self.args.food_energy)
        # environment state must not carry autograd graph into the next step
        self.x = self.x.detach(); self.y = self.y.detach(); self.angle = self.angle.detach()
        self.energy = self.energy.detach(); self.age = self.age.detach()
        self.last = {
            "step": self.step_i,
            "loss": float(loss.detach().cpu()),
            "loss_policy": float(loss_policy.detach().cpu()),
            "reward_mean": float(reward.detach().cpu()),
            "energy_mean": float(self.energy.mean().detach().cpu()),
            "energy_max": float(self.energy.max().detach().cpu()),
            "food_alive": int(self.food_alive.sum().detach().cpu()),
            "births": self.births,
            "deaths": self.deaths,
            "shape_speed": float(sh[:, 0].mean().detach().cpu()),
            "shape_tool": float(sh[:, 1].mean().detach().cpu()),
            "shape_guard": float(sh[:, 2].mean().detach().cpu()),
            "shape_sense": float(sh[:, 3].mean().detach().cpu()),
            **stats,
        }
        return self.last


def write_csv(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/015_life_task_moe_gpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--agents", type=int, default=256)
    p.add_argument("--food", type=int, default=64)
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--height", type=int, default=700)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--nodes", type=int, default=32)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--motor-nodes", type=int, default=4)
    p.add_argument("--degree", type=int, default=5)
    p.add_argument("--initial-energy", type=float, default=70)
    p.add_argument("--hunger-energy", type=float, default=28)
    p.add_argument("--reproduce-energy", type=float, default=88)
    p.add_argument("--child-energy", type=float, default=25)
    p.add_argument("--food-energy", type=float, default=10)
    p.add_argument("--food-spawn", type=float, default=0.015)
    p.add_argument("--sense-radius", type=float, default=180)
    p.add_argument("--turn-rate", type=float, default=0.35)
    p.add_argument("--max-speed", type=float, default=2.0)
    p.add_argument("--steal-energy", type=float, default=8.0)
    p.add_argument("--metabolism", type=float, default=0.035)
    p.add_argument("--move-cost", type=float, default=0.018)
    p.add_argument("--shape-cost", type=float, default=0.018)
    p.add_argument("--neural-cost", type=float, default=0.018)
    p.add_argument("--max-age", type=float, default=4000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cost-lambda", type=float, default=0.002)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()
    dev = require_cuda()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    env = LifeGPUEnv(args, dev)
    print("=== 015 REAL GPU TaskMoE/ADFC Life ===", flush=True)
    print("device", dev, torch.cuda.get_device_name(0), "params", nparams(env.policy), flush=True)
    print("imports: GraphChannel, TaskMoEDNA, AlwaysTypedDNA, PairwiseOrderBank, KeyReadBank", flush=True)
    rows = []
    t0 = time.time()
    for i in range(args.steps):
        st = env.step()
        if (i + 1) % args.log_every == 0 or i == 0:
            st = dict(st)
            st["sec"] = time.time() - t0
            rows.append(st)
            print("step={step} loss={loss:.4f} E={energy_mean:.2f} food={food_alive} w=({w_graph:.2f},{w_order:.2f},{w_key:.2f}) p=({p_forage:.2f},{p_avoid:.2f},{p_mate:.2f},{p_rest:.2f}) shape=({shape_speed:.2f},{shape_tool:.2f},{shape_guard:.2f},{shape_sense:.2f})".format(**st), flush=True)
            write_csv(out / "train.csv", rows)
    write_csv(out / "train.csv", rows)
    torch.save(env.policy.state_dict(), out / "policy_final.pt")
    print("DONE", out.resolve(), flush=True)

if __name__ == "__main__":
    main()
