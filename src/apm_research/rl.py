"""E4 learned systems: B3 (APM-L learned regulator) and B4 (capacity-matched recurrent policy), trained with
recurrent PPO (MASTER_PREREGISTRATION.md §1.2, §2 E4).

Both receive the same per-decision input: the shared decision features (`features.WIDTH`, which include the
explicit memory expectations) and the shared consequence predictor's outputs for every action (P_col, P_slip,
progress), 41 + 12 = 53 numbers. Both keep a recurrent state across the episodes of a lifetime.

* B3: input → fast GRU latent + slow leaky latent (learned per-unit rates, initialised slow) → bounded preference
  `β·tanh(·)` added to B1's transparent utility; action logits = (utility + preference) / τ.
  Ablations: no explicit memory features (memory inputs zeroed), no persistent latent (reset every episode),
  fast-only, slow-only; evaluation-only negative controls: identity-shuffled memory, history reset every decision.
* B4: input → GRU → logits. Hidden size chosen so that parameters are within ±10% of B3 (`b4_hidden`).

The safety supervisor, actions and observations are the shared ones; nothing here touches the simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from apm_research.features import MEM_CONF, MEM_P, WIDTH
from apm_research.policies import FeaturePolicy
from apm_testbeds.wheeled_ctrl import N_ACTIONS

IN = WIDTH + 3 * N_ACTIONS
REWARD = {"goal_reached": 1.0, "collision": -1.0, "severe": -2.0, "slip": -0.25}  # = tuning utility J (§4.3)


@dataclass
class B3Config:
    fast: int = 32
    slow: int = 16
    beta: float = 0.5        # preference bound (progress units are ≈ 0–0.6 per 2 s)
    tau: float = 0.05        # logit temperature
    memory_features: bool = True
    persistent: bool = True  # latent carried across episodes of a lifetime


class B3Net(nn.Module):
    def __init__(self, cfg: B3Config):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(IN)
        self.enc = nn.Sequential(nn.Linear(IN, 64), nn.Tanh())
        self.fast = nn.GRUCell(64, cfg.fast) if cfg.fast else None
        if cfg.slow:
            self.slow_in = nn.Linear(64 + cfg.fast, cfg.slow)
            self.slow_rate = nn.Parameter(torch.full((cfg.slow,), -4.0))  # sigmoid(−4) ≈ 0.018 per decision
        latent = cfg.fast + cfg.slow
        self.pref = nn.Linear(latent, N_ACTIONS)
        nn.init.zeros_(self.pref.weight)
        nn.init.zeros_(self.pref.bias)  # starts exactly at B1's rule
        self.value = nn.Linear(latent + 64, 1)

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.fast + self.cfg.slow)

    def forward(self, x: torch.Tensor, utility: torch.Tensor, state: torch.Tensor):
        e = self.enc(self.norm(x))
        f, s = state[:, :self.cfg.fast], state[:, self.cfg.fast:]
        if self.fast is not None:
            f = self.fast(e, f)
        if self.cfg.slow:
            rate = torch.sigmoid(self.slow_rate)
            s = s + rate * (torch.tanh(self.slow_in(torch.cat([e, f], 1))) - s)
        h = torch.cat([f, s], 1)
        logits = (utility + self.cfg.beta * torch.tanh(self.pref(h))) / self.cfg.tau
        return logits, self.value(torch.cat([h, e], 1)).squeeze(1), h


class B4Net(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.norm = nn.LayerNorm(IN)
        self.enc = nn.Sequential(nn.Linear(IN, 64), nn.Tanh())
        self.gru = nn.GRUCell(64, hidden)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.hidden)

    def forward(self, x: torch.Tensor, utility: torch.Tensor, state: torch.Tensor):
        h = self.gru(self.enc(self.norm(x)), state)
        return self.pi(h), self.value(h).squeeze(1), h


def parameters(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def b4_hidden(target: int, scale: float = 1.0) -> int:
    """Smallest GRU hidden size whose B4 parameter count reaches `scale × target`."""
    for h in range(4, 1024):
        if parameters(B4Net(h)) >= scale * target:
            return h
    raise ValueError(target)


def make_net(spec: dict) -> nn.Module:
    if spec["kind"] == "B3":
        return B3Net(B3Config(**spec.get("b3", {})))
    return B4Net(spec["hidden"])


class LearnedPolicy(FeaturePolicy):
    """Runs a B3/B4 network in the shared harness. `sample=True` during training (records the trajectory),
    greedy at evaluation. Evaluation-only controls: `shuffle_identity` (memory inputs permuted across the tracked
    people each decision), `reset_every_decision` (history reset)."""

    def __init__(self, net: nn.Module, predictor, lam_c: float, lam_s: float, sample: bool = False,
                 seed: int | np.random.SeedSequence = 0, persistent: bool = True, memory_features: bool = True,
                 shuffle_identity: bool = False, reset_every_decision: bool = False, name: str = "learned"):
        self.net, self.predictor, self.lam_c, self.lam_s = net, predictor, lam_c, lam_s
        self.sample, self.rng = sample, np.random.default_rng(seed)
        self.persistent, self.memory_features = persistent, memory_features
        self.shuffle_identity, self.reset_every_decision = shuffle_identity, reset_every_decision
        self.name = name
        self.trajectory: list[tuple] = []

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.h = self.net.init_state()
        self.trajectory = []
        self.new_episode = True

    def begin_episode(self, t0: float) -> None:
        super().begin_episode(t0)
        self.new_episode = True
        if not self.persistent:
            self.h = self.net.init_state()

    def inputs(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        x = self.features(t).copy()
        if self.shuffle_identity:
            order = self.rng.permutation(len(MEM_P))
            x[MEM_P], x[MEM_CONF] = x[MEM_P][order], x[MEM_CONF][order]
        if not self.memory_features:
            x[MEM_P], x[MEM_CONF] = 0.0, 0.0
        p_col, p_slip, progress = self.predictor.predict_actions(x)
        utility = progress - self.lam_c * p_col - self.lam_s * p_slip
        return np.concatenate([x, p_col, p_slip, progress]).astype(np.float32), utility.astype(np.float32)

    @torch.no_grad()
    def decide(self, t: float) -> int:
        z, u = self.inputs(t)
        reset = (self.new_episode and not self.persistent) or self.reset_every_decision
        self.new_episode = False
        if self.reset_every_decision:
            self.h = self.net.init_state()
        logits, value, h = self.net(torch.from_numpy(z)[None], torch.from_numpy(u)[None], self.h)
        self.h = h
        if self.sample:
            p = torch.softmax(logits[0], 0).numpy().astype(np.float64)
            p /= p.sum()
            a = int(self.rng.choice(N_ACTIONS, p=p))
            self.trajectory.append((z, u, a, float(math.log(max(p[a], 1e-12))), float(value[0]), reset))
            return a
        return int(torch.argmax(logits[0]))


def period_reward(events) -> float:
    r = 0.0
    for ev in events:
        if ev[0] == "goal_reached":
            r += REWARD["goal_reached"]
        elif ev[0] == "collision":
            r += REWARD["collision"] + (REWARD["severe"] if ev[2] else 0.0)
        elif ev[0] == "slip":
            r += REWARD["slip"]
    return r


class RewardProbe:
    """Wraps a LearnedPolicy in the shared harness and accumulates the reward of each decision period."""

    def __init__(self, policy: LearnedPolicy):
        self.policy = policy
        self.rewards: list[float] = []
        self.pending = 0.0
        self.started = False

    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        self.pending += period_reward(obs.events)
        self.policy.observe(obs, t)

    def decide(self, t: float) -> int:
        if self.started:
            self.rewards.append(self.pending)
        self.pending, self.started = 0.0, True
        return self.policy.decide(t)

    def end_episode(self, t: float) -> None:
        self.rewards.append(self.pending)
        self.pending, self.started = 0.0, False
        self.policy.end_episode(t)


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95):
    adv = np.zeros_like(rewards)
    last = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        nxt = values[i + 1] if i + 1 < len(values) else 0.0
        delta = rewards[i] + gamma * nxt - values[i]
        last = delta + gamma * lam * last
        adv[i] = last
    return adv, adv + values


def ppo_update(net: nn.Module, opt: torch.optim.Optimizer, batch: list[dict], epochs: int = 4, clip: float = 0.2,
               ent: float = 0.01, vf: float = 0.5, max_norm: float = 0.5) -> dict:
    """Full-sequence recurrent PPO over lifetimes (all lifetimes in a batch have equal length)."""
    Z = torch.from_numpy(np.stack([b["z"] for b in batch]))       # [B, T, IN]
    U = torch.from_numpy(np.stack([b["u"] for b in batch]))       # [B, T, A]
    A = torch.from_numpy(np.stack([b["a"] for b in batch]))       # [B, T]
    LP = torch.from_numpy(np.stack([b["logp"] for b in batch]))
    ADV = torch.from_numpy(np.stack([b["adv"] for b in batch]))
    RET = torch.from_numpy(np.stack([b["ret"] for b in batch]))
    RESET = torch.from_numpy(np.stack([b["reset"] for b in batch]))  # [B, T] state reset before step
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    stats = {}
    for _ in range(epochs):
        h = net.init_state(Z.shape[0])
        logits_l, values_l = [], []
        for t in range(Z.shape[1]):
            h = torch.where(RESET[:, t:t + 1], net.init_state(Z.shape[0]), h)
            lg, v, h = net(Z[:, t], U[:, t], h)
            logits_l.append(lg)
            values_l.append(v)
        logits, values = torch.stack(logits_l, 1), torch.stack(values_l, 1)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(A)
        ratio = torch.exp(logp - LP)
        pg = -torch.min(ratio * ADV, torch.clamp(ratio, 1 - clip, 1 + clip) * ADV).mean()
        vloss = ((values - RET) ** 2).mean()
        entropy = dist.entropy().mean()
        loss = pg + vf * vloss - ent * entropy
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        opt.step()
        stats = {"pg": float(pg), "v": float(vloss), "entropy": float(entropy),
                 "clipfrac": float(((ratio - 1).abs() > clip).float().mean())}
    return stats
