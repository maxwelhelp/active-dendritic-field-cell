#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from graph_adfc_worm import require_cuda, nparams, wcsv
from graph_adfc_worm_typed_bank import PairwiseOrderBank, KeyReadBank
from genome_task_moe import GraphChannel, TaskMoEDNA, AlwaysTypedDNA

LIFE_TASKS = ["forage", "avoid", "pair", "rest"]
ACTIONS = 13
FDIM = 29


def detach_nonparam_tensor_state(module: nn.Module):
    for m in module.modules():
        ps = set(m._parameters.keys())
        bs = set(m._buffers.keys())
        for name, val in list(vars(m).items()):
            if name in ps or name in bs:
                continue
            if torch.is_tensor(val):
                try:
                    setattr(m, name, val.detach())
                except Exception:
                    pass


class LifeTaskMoEPolicy(nn.Module):
    def __init__(self, nodes=32, dim=32, seq_len=32, motor_nodes=4, degree=5):
        super().__init__()
        self.graph = GraphChannel(nodes, dim, FDIM, 4, motor_nodes, degree)
        self.order = PairwiseOrderBank(FDIM, dim)
        self.key = KeyReadBank(FDIM, dim)
        self.task_router = nn.Sequential(nn.LayerNorm(3 * FDIM), nn.Linear(3 * FDIM, 96), nn.GELU(), nn.Linear(96, 4))
        self.graph_to_action = nn.Sequential(nn.Linear(2, 48), nn.GELU(), nn.Linear(48, ACTIONS))
        self.order_to_action = nn.Linear(dim, ACTIONS)
        self.key_to_action = nn.Linear(dim, ACTIONS)
        self.channel_by_task = nn.Parameter(torch.tensor([
            [0.50, 0.15, 0.35],
            [0.45, 0.40, 0.15],
            [0.25, 0.35, 0.40],
            [0.55, 0.25, 0.20],
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
            oa = self.order.last_abs.detach() if torch.is_tensor(self.order.last_abs) else torch.tensor(float(self.order.last_abs), device=x.device)
            ke = self.key.last_entropy.detach() if torch.is_tensor(self.key.last_entropy) else torch.tensor(float(self.key.last_entropy), device=x.device)
            self.last_stats = {
                "p_forage": float(tm[0]), "p_avoid": float(tm[1]), "p_pair": float(tm[2]), "p_rest": float(tm[3]),
                "w_graph": float(wm[0]), "w_order": float(wm[1]), "w_key": float(wm[2]),
                "order_abs": float(oa.cpu()), "key_entropy": float(ke.cpu()),
            }
            self.order.last_abs = oa.detach()
            self.key.last_entropy = ke.detach()
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
        self.phase = torch.rand(self.N, device=device) * math.tau
        self.seg_n = int(args.segments)
        self.seg_len = float(args.segment_len)
        k = torch.arange(self.seg_n, device=device).float()[None, :]
        self.seg_x = (self.x[:, None] - torch.cos(self.angle)[:, None] * self.seg_len * k) % self.W
        self.seg_y = (self.y[:, None] - torch.sin(self.angle)[:, None] * self.seg_len * k) % self.H
        self.energy = torch.full((self.N,), args.initial_energy, device=device)
        self.age = torch.zeros(self.N, device=device)
        self.sex = torch.randint(0, 2, (self.N,), device=device).float()
        self.agent_code = torch.randn(self.N, 4, device=device)
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

    def smell_food(self):
        dx = self.wrap_delta(self.food_x[None, :] - self.x[:, None], self.W)
        dy = self.wrap_delta(self.food_y[None, :] - self.y[:, None], self.H)
        d = torch.sqrt((dx * dx + dy * dy).clamp_min(1e-6))
        w = torch.exp(-d / self.args.odor_radius) * self.food_alive[None, :].float()
        s = w.sum(dim=1).clamp_max(8.0) / 8.0
        vx = (w * dx / (d + 1e-6)).sum(dim=1) / (w.sum(dim=1) + 1e-6)
        vy = (w * dy / (d + 1e-6)).sum(dim=1) / (w.sum(dim=1) + 1e-6)
        return s, vx.clamp(-1, 1), vy.clamp(-1, 1)

    def smell_agents(self, sex_value: float):
        dx = self.wrap_delta(self.x[None, :] - self.x[:, None], self.W)
        dy = self.wrap_delta(self.y[None, :] - self.y[:, None], self.H)
        d = torch.sqrt((dx * dx + dy * dy).clamp_min(1e-6))
        mask = (self.sex[None, :] == sex_value).float()
        eye = torch.eye(self.N, device=self.device)
        mask = mask * (1.0 - eye)
        w = torch.exp(-d / self.args.odor_radius) * mask
        s = w.sum(dim=1).clamp_max(8.0) / 8.0
        vx = (w * dx / (d + 1e-6)).sum(dim=1) / (w.sum(dim=1) + 1e-6)
        vy = (w * dy / (d + 1e-6)).sum(dim=1) / (w.sum(dim=1) + 1e-6)
        return s, vx.clamp(-1, 1), vy.clamp(-1, 1)

    def encode_obs_sequence(self):
        fidx, fdx, fdy, fd = self.nearest_food()
        aidx, adx, ady, ad = self.nearest_agent()
        fs, fvx, fvy = self.smell_food()
        ms, mvx, mvy = self.smell_agents(0.0)
        qs, qvx, qvy = self.smell_agents(1.0)
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
        hunger = base[:, 1]
        stronger = (self.energy[aidx] > self.energy * 1.12).float()
        danger = base[:, 9] * stronger
        can_pair = (self.energy > self.args.reproduce_energy).float()
        opposite_smell = torch.where(self.sex > 0.5, ms, qs)
        rest_need = (self.energy < self.args.hunger_energy * 0.55).float()
        task_score = torch.stack([hunger + fs, danger, can_pair * opposite_smell, rest_need], dim=1)
        tid = task_score.argmax(dim=1)
        base[torch.arange(self.N, device=self.device), 12 + tid] = 1.0
        base[:, 16:20] = self.agent_code
        base[:, 20] = fs; base[:, 21] = fvx; base[:, 22] = fvy
        base[:, 23] = ms; base[:, 24] = mvx; base[:, 25] = mvy
        base[:, 26] = qs; base[:, 27] = qvx; base[:, 28] = qvy
        x = base[:, None, :].repeat(1, self.seq_len, 1)
        pos = torch.linspace(-1, 1, self.seq_len, device=self.device)[None, :, None]
        x[:, :, 0:1] = x[:, :, 0:1] + 0.05 * pos
        return x, tid, (fidx, fdx, fdy, fd, aidx, adx, ady, ad, fvx, fvy, mvx, mvy, qvx, qvy)

    def teacher_targets(self, aux):
        fidx, fdx, fdy, fd, aidx, adx, ady, ad, fvx, fvy, mvx, mvy, qvx, qvy = aux
        target = torch.zeros(self.N, ACTIONS, device=self.device) + 0.20
        target[:, 6] = 0.20
        hunger = 1.0 - (self.energy / self.args.hunger_energy).clamp(0, 1)
        desired = torch.atan2(fvy + 0.55 * fdy / self.args.sense_radius, fvx + 0.55 * fdx / self.args.sense_radius)
        diff = (desired - self.angle + math.pi) % (2 * math.pi) - math.pi
        target[:, 0] = (diff < -0.05).float().clamp(0, 1)
        target[:, 1] = (diff > 0.05).float().clamp(0, 1)
        target[:, 2] = (0.20 + 0.70 * hunger).clamp(0, 1)
        near_food = (1.0 - fd / self.args.sense_radius).clamp(0, 1)
        target[:, 3] = near_food
        stronger = (self.energy[aidx] > self.energy * 1.15).float()
        near_agent = (1.0 - ad / self.args.sense_radius).clamp(0, 1)
        target[:, 4] = ((1.0 - stronger) * near_agent * (0.20 + 0.60 * hunger)).clamp(0, 1)
        opposite_vx = torch.where(self.sex > 0.5, mvx, qvx)
        opposite_vy = torch.where(self.sex > 0.5, mvy, qvy)
        pair_dir = torch.atan2(opposite_vy, opposite_vx)
        pair_diff = (pair_dir - self.angle + math.pi) % (2 * math.pi) - math.pi
        can_pair = (self.energy > self.args.reproduce_energy).float()
        target[:, 5] = can_pair * near_agent
        pair_mask = can_pair > 0.5
        target[pair_mask, 0] = torch.maximum(target[pair_mask, 0], (pair_diff[pair_mask] < -0.05).float())
        target[pair_mask, 1] = torch.maximum(target[pair_mask, 1], (pair_diff[pair_mask] > 0.05).float())
        target[:, 6] = ((self.energy < self.args.hunger_energy * 0.55).float() * (1.0 - near_food)).clamp(0, 1)
        target[:, 7] = (0.15 + 0.60 * hunger).clamp(0, 1)
        target[:, 8] = (0.25 + 0.55 * near_food).clamp(0, 1)
        target[:, 9] = (0.20 + 0.60 * stronger * near_agent).clamp(0, 1)
        target[:, 10] = (0.25 + 0.55 * (1.0 - near_food)).clamp(0, 1)
        target[:, 11] = (0.5 + 0.5 * torch.sin(self.phase)).clamp(0, 1)
        target[:, 12] = (0.45 + 0.25 * near_agent).clamp(0, 1)
        return target

    def step(self):
        self.step_i += 1
        with torch.no_grad():
            xseq, tid, aux = self.encode_obs_sequence()
        xseq = xseq.detach()
        logits = self.policy(xseq)
        action = torch.sigmoid(logits)
        target = self.teacher_targets(aux)
        per_agent_mse = ((action - target) ** 2).mean(dim=1)
        hunger_now = (1.0 - (self.energy / self.args.hunger_energy).clamp(0, 1)).detach()
        risk_now = (self.energy < self.args.hunger_energy * 0.45).float().detach()
        # Survival auditor / critic: prioritize local correction for agents at risk.
        # This avoids treating agent-vs-agent transfer as one global punishment.
        survival_weight = (1.0 + self.args.hunger_loss_weight * hunger_now + self.args.risk_loss_weight * risk_now).clamp(1.0, 5.0)
        loss_policy = (per_agent_mse * survival_weight).mean()
        tp_for_loss = self.policy.task_probs(xseq).clamp_min(1e-8)
        loss_task = F.nll_loss(tp_for_loss.log(), tid)
        stats = self.policy.stats()
        detach_nonparam_tensor_state(self.policy)
        reg = self.args.cost_lambda * action.mean()
        loss = loss_policy + self.args.task_lambda * loss_task + reg
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        action_env = action.detach()
        bend = (action_env[:, 1] - action_env[:, 0])
        wave = action_env[:, 2]
        mouth = action_env[:, 3]
        contact = action_env[:, 4]
        pair = action_env[:, 5]
        rest = action_env[:, 6]
        sh = action_env[:, 7:11].clamp_min(1e-4)
        sh = sh / sh.sum(dim=1, keepdim=True)
        phase_drive = action_env[:, 11]
        stiffness = action_env[:, 12]
        old_energy = self.energy.clone()
        self.phase = (self.phase + 0.20 + 0.55 * phase_drive + 0.30 * wave) % math.tau
        body_wave = torch.sin(self.phase) * wave * (0.35 + 1.15 * stiffness)
        self.angle = self.angle + self.args.turn_rate * (0.55 * bend + 0.45 * body_wave) * (1.0 - 0.40 * rest)
        k = torch.arange(self.seg_n, device=self.device).float()[None, :]
        seg_phase = self.phase[:, None] - k * self.args.segment_phase_lag
        local_curve = torch.sin(seg_phase) * wave[:, None] * (0.25 + 1.20 * stiffness[:, None])
        seg_ang = self.angle[:, None] + local_curve
        # Undulatory locomotion: no direct teleport-thrust; forward motion comes from wave/stiffness
        # and is damped by lateral friction. More wave costs energy below.
        undulation = torch.relu(wave * (0.25 + stiffness) * (0.65 + 0.35 * torch.cos(body_wave - bend)))
        speed = self.args.max_speed * self.speed * (0.20 + 1.60 * sh[:, 0]) * undulation * (1.0 - 0.55 * rest)
        dx_raw = torch.cos(self.angle) * speed
        dy_raw = torch.sin(self.angle) * speed
        # anisotropic friction: sideways component slips less than forward component
        tx = torch.cos(self.angle); ty = torch.sin(self.angle)
        forward = dx_raw * tx + dy_raw * ty
        latx = dx_raw - forward * tx; laty = dy_raw - forward * ty
        dx = forward * tx + self.args.lateral_slip * latx
        dy = forward * ty + self.args.lateral_slip * laty
        head_x = (self.seg_x[:, 0] + dx) % self.W
        head_y = (self.seg_y[:, 0] + dy) % self.H
        new_xs = [head_x]
        new_ys = [head_y]
        prev_x = head_x; prev_y = head_y
        for j in range(1, self.seg_n):
            # target joint follows previous segment at fixed length and learned curvature angle
            px = (prev_x - torch.cos(seg_ang[:, j]) * self.seg_len) % self.W
            py = (prev_y - torch.sin(seg_ang[:, j]) * self.seg_len) % self.H
            # friction/muscle inertia: segments do not instantly teleport to target
            oldx = self.seg_x[:, j]; oldy = self.seg_y[:, j]
            nx = (oldx * self.args.segment_inertia + px * (1.0 - self.args.segment_inertia)) % self.W
            ny = (oldy * self.args.segment_inertia + py * (1.0 - self.args.segment_inertia)) % self.H
            # length projection back to fixed distance from previous joint
            vx = self.wrap_delta(nx - prev_x, self.W)
            vy = self.wrap_delta(ny - prev_y, self.H)
            d = torch.sqrt((vx * vx + vy * vy).clamp_min(1e-6))
            nx = (prev_x + vx / d * self.seg_len) % self.W
            ny = (prev_y + vy / d * self.seg_len) % self.H
            new_xs.append(nx); new_ys.append(ny)
            prev_x = nx; prev_y = ny
        self.seg_x = torch.stack(new_xs, dim=1).detach()
        self.seg_y = torch.stack(new_ys, dim=1).detach()
        self.x = self.seg_x[:, 0]
        self.y = self.seg_y[:, 0]
        seg_dx = self.wrap_delta(self.seg_x[:, 1:] - self.seg_x[:, :-1], self.W)
        seg_dy = self.wrap_delta(self.seg_y[:, 1:] - self.seg_y[:, :-1], self.H)
        stretch_error = (torch.sqrt((seg_dx * seg_dx + seg_dy * seg_dy).clamp_min(1e-6)) - self.seg_len).abs().mean()
        fidx, fdx, fdy, fd = self.nearest_food()
        aidx, adx, ady, ad = self.nearest_agent()
        near = fd < (5.0 + 8.0 * sh[:, 1] * self.tool)
        gain = near.float() * mouth * self.args.food_energy
        self.energy = self.energy + gain
        self.food_alive[fidx[near]] = False
        dead_food = ~self.food_alive
        respawn = dead_food & (torch.rand(self.F, device=self.device) < self.args.food_spawn)
        rn = int(respawn.sum().item())
        if rn > 0:
            self.food_x[respawn] = torch.rand(rn, device=self.device) * self.W
            self.food_y[respawn] = torch.rand(rn, device=self.device) * self.H
        self.food_alive[respawn] = True
        near_a = ad < (6.0 + 7.0 * sh[:, 1] * self.tool)
        transfer = near_a.float() * contact * self.args.transfer_energy * (0.45 + sh[:, 1]) / (1.0 + self.guard[aidx] * sh[aidx, 2])
        transfer = torch.minimum(transfer, self.energy[aidx].clamp_min(0))
        contact_events = int((transfer > 1e-5).sum().detach().cpu())
        transfer_mean = float(transfer.mean().detach().cpu())
        self.energy = self.energy + transfer * 0.75
        self.energy[aidx] = self.energy[aidx] - transfer
        can_birth = (pair > 0.65) & (self.energy > self.args.reproduce_energy)
        births = int(can_birth.sum().item())
        if births:
            self.births += births
            self.energy[can_birth] -= self.args.child_energy * 0.5
        cost = self.args.metabolism + self.args.move_cost * speed
        cost = cost + self.args.shape_cost * (sh[:, 0] * self.speed + sh[:, 1] * self.tool + sh[:, 2] * self.guard + sh[:, 3])
        cost = cost + self.args.wiggle_cost * (wave.abs() + bend.abs() + phase_drive.abs() + stiffness.abs())
        cost = cost + self.args.stretch_cost * stretch_error
        cost = cost + self.args.neural_cost * action_env.abs().mean(dim=1)
        self.energy = self.energy - cost
        self.age = self.age + 1
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
                self.phase[dead] = torch.rand(dead_n, device=self.device) * math.tau
                kk = torch.arange(self.seg_n, device=self.device).float()[None, :]
                self.seg_x[dead] = (self.x[dead, None] - torch.cos(self.angle[dead])[:, None] * self.seg_len * kk) % self.W
                self.seg_y[dead] = (self.y[dead, None] - torch.sin(self.angle[dead])[:, None] * self.seg_len * kk) % self.H
                self.agent_code[dead] = torch.randn(dead_n, 4, device=self.device)
        reward = (self.energy - old_energy).mean() / max(1.0, self.args.food_energy)
        self.x = self.x.detach(); self.y = self.y.detach(); self.angle = self.angle.detach(); self.phase = self.phase.detach()
        self.seg_x = self.seg_x.detach(); self.seg_y = self.seg_y.detach()
        self.energy = self.energy.detach(); self.age = self.age.detach()
        self.last = {
            "step": self.step_i,
            "loss": float(loss.detach().cpu()),
            "loss_policy": float(loss_policy.detach().cpu()),
            "loss_task": float(loss_task.detach().cpu()),
            "survival_weight_mean": float(survival_weight.mean().detach().cpu()),
            "hunger_mean": float(hunger_now.mean().detach().cpu()),
            "death_events": dead_n,
            "reward_mean": float(reward.detach().cpu()),
            "energy_mean": float(self.energy.mean().detach().cpu()),
            "energy_max": float(self.energy.max().detach().cpu()),
            "food_alive": int(self.food_alive.sum().detach().cpu()),
            "births": self.births,
            "deaths": self.deaths,
            "contact_events": contact_events,
            "transfer_mean": transfer_mean,
            "shape_speed": float(sh[:, 0].mean().detach().cpu()),
            "shape_tool": float(sh[:, 1].mean().detach().cpu()),
            "shape_guard": float(sh[:, 2].mean().detach().cpu()),
            "shape_sense": float(sh[:, 3].mean().detach().cpu()),
            "wave_mean": float(wave.mean().detach().cpu()),
            "bend_mean": float(bend.abs().mean().detach().cpu()),
            "speed_mean": float(speed.mean().detach().cpu()),
            "stretch_error": float(stretch_error.detach().cpu()),
            "segments": self.seg_n,
            **stats,
        }
        return self.last


def write_csv(path, rows):
    if not rows:
        return
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
    p.add_argument("--odor-radius", type=float, default=145)
    p.add_argument("--sense-radius", type=float, default=180)
    p.add_argument("--turn-rate", type=float, default=0.35)
    p.add_argument("--max-speed", type=float, default=3.2)
    p.add_argument("--segments", type=int, default=7)
    p.add_argument("--segment-len", type=float, default=4.0)
    p.add_argument("--segment-phase-lag", type=float, default=0.85)
    p.add_argument("--segment-inertia", type=float, default=0.38)
    p.add_argument("--lateral-slip", type=float, default=0.18)
    p.add_argument("--transfer-energy", type=float, default=8.0)
    p.add_argument("--metabolism", type=float, default=0.035)
    p.add_argument("--move-cost", type=float, default=0.018)
    p.add_argument("--shape-cost", type=float, default=0.018)
    p.add_argument("--wiggle-cost", type=float, default=0.012)
    p.add_argument("--stretch-cost", type=float, default=0.020)
    p.add_argument("--neural-cost", type=float, default=0.018)
    p.add_argument("--max-age", type=float, default=4000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cost-lambda", type=float, default=0.002)
    p.add_argument("--task-lambda", type=float, default=0.05)
    p.add_argument("--hunger-loss-weight", type=float, default=1.5)
    p.add_argument("--risk-loss-weight", type=float, default=2.0)
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
            st = dict(st); st["sec"] = time.time() - t0; rows.append(st)
            print("step={step} loss={loss:.4f} E={energy_mean:.2f} food={food_alive} contacts={contact_events} transfer={transfer_mean:.3f} w=({w_graph:.2f},{w_order:.2f},{w_key:.2f}) p=({p_forage:.2f},{p_avoid:.2f},{p_pair:.2f},{p_rest:.2f}) shape=({shape_speed:.2f},{shape_tool:.2f},{shape_guard:.2f},{shape_sense:.2f}) wave={wave_mean:.2f} speed={speed_mean:.2f} hunger={hunger_mean:.2f} sw={survival_weight_mean:.2f}".format(**st), flush=True)
            write_csv(out / "train.csv", rows)
    write_csv(out / "train.csv", rows)
    torch.save(env.policy.state_dict(), out / "policy_final.pt")
    print("DONE", out.resolve(), flush=True)

if __name__ == "__main__":
    main()
