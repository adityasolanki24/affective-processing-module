"""G4/G5 learned policies and training methods, built on programme F's networks (`apm_research.rl`).

`Learner` runs a network (F's B3Net = bounded learned residual on a utility, or B4Net = ordinary GRU actor-critic) in
the shared harness. It always carries a *shadow teacher*: an A0 rule policy (`apm_select.regulators.Regulated`, i.e.
frozen predictor + memory + the G1-selected penalty) that receives the identical observation stream. Because the
teacher's memory, features and regulator are driven by observations only, its action at every state the learner
visits is exact, which gives:
  * the A0 utility used as the residual anchor (A4: logits = (u_A0 + β·tanh(pref))/τ, starts at A0 when pref = 0),
  * teacher labels for behaviour cloning and DAgger.
Network inputs are F's: 41 decision features ⊕ predictor outputs for each action (P_col, P_slip, progress) = 53;
`memory_features=False` zeroes the explicit-memory columns (A2 = ordinary GRU without explicit memory).

Training methods (scripts/g45_train.py): PPO from scratch; BC on teacher rollouts; BC → PPO; DAgger (β-mixture
rollouts, every visited state labelled by the teacher); every environment interaction is counted, teacher rollouts
included.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from apm_research.features import MEM_CONF, MEM_P
from apm_research.rl import IN, period_reward  # noqa: F401  (re-exported for scripts)
from apm_select.regulators import Regulated, make_regulator
from apm_testbeds.wheeled_ctrl import N_ACTIONS


def teacher_from_spec(predictor, spec: dict) -> Regulated:
    return Regulated(predictor, spec["lam_c"], spec["lam_s"], make_regulator(spec["regulator"],
                                                                              spec.get("params", {})), name="A0")


class Learner:
    """mode: "greedy" (evaluation), "sample" (PPO rollouts), "teacher" (execute the teacher, record its labels: BC
    data), "dagger" (execute teacher with probability beta, else the learner's greedy action; record labels)."""

    def __init__(self, net, teacher: Regulated, mode: str = "greedy", memory_features: bool = True,
                 persistent: bool = True, beta: float = 0.0, seed: int | np.random.SeedSequence = 0,
                 name: str = "learned"):
        self.net, self.teacher, self.mode = net, teacher, mode
        self.memory_features, self.persistent, self.beta = memory_features, persistent, beta
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.trajectory: list[tuple] = []   # PPO: (z, u, a, logp, value, reset)
        self.labelled: list[tuple] = []     # BC/DAgger: (z, u, teacher action, reset)

    def begin_lifetime(self) -> None:
        self.teacher.begin_lifetime()
        self.h = self.net.init_state()
        self.trajectory, self.labelled = [], []
        self.new_episode = True

    def begin_episode(self, t0: float) -> None:
        self.teacher.begin_episode(t0)
        self.new_episode = True
        if not self.persistent:
            self.h = self.net.init_state()

    def observe(self, obs, t: float) -> None:
        self.teacher.observe(obs, t)

    def end_episode(self, t: float) -> None:
        self.teacher.end_episode(t)

    def _inputs(self, t: float) -> tuple[np.ndarray, np.ndarray, int]:
        teacher_action = self.teacher.decide(t)        # computes features, predictor outputs and A0 utility
        x = self.teacher.last_features.copy()
        if not self.memory_features:
            x[MEM_P], x[MEM_CONF] = 0.0, 0.0
        p_col, p_slip, progress = self.teacher.predictor.predict_actions(x)
        z = np.concatenate([x, p_col, p_slip, progress]).astype(np.float32)
        return z, self.teacher.last_utility.astype(np.float32), teacher_action

    @torch.no_grad()
    def decide(self, t: float) -> int:
        z, u, ta = self._inputs(t)
        reset = self.new_episode and not self.persistent
        self.new_episode = False
        logits, value, h = self.net(torch.from_numpy(z)[None], torch.from_numpy(u)[None], self.h)
        self.h = h
        if self.mode == "sample":
            p = torch.softmax(logits[0], 0).numpy().astype(np.float64)
            p /= p.sum()
            a = int(self.rng.choice(N_ACTIONS, p=p))
            self.trajectory.append((z, u, a, float(math.log(max(p[a], 1e-12))), float(value[0]), reset))
            return a
        if self.mode in ("teacher", "dagger"):
            self.labelled.append((z, u, ta, reset))
            if self.mode == "teacher" or self.rng.random() < self.beta:
                return ta
        return int(torch.argmax(logits[0]))


def bc_loss(net, batch: list[dict]) -> torch.Tensor:
    """Cross-entropy of the teacher's actions along full lifetimes (recurrent, equal lengths within a batch)."""
    Z = torch.from_numpy(np.stack([b["z"] for b in batch]))
    U = torch.from_numpy(np.stack([b["u"] for b in batch]))
    A = torch.from_numpy(np.stack([b["a"] for b in batch]))
    R = torch.from_numpy(np.stack([b["reset"] for b in batch]))
    h = net.init_state(Z.shape[0])
    logits = []
    for t in range(Z.shape[1]):
        h = torch.where(R[:, t:t + 1], net.init_state(Z.shape[0]), h)
        lg, _, h = net(Z[:, t], U[:, t], h)
        logits.append(lg)
    return torch.nn.functional.cross_entropy(torch.stack(logits, 1).reshape(-1, N_ACTIONS), A.reshape(-1))


def labelled_arrays(policy: Learner) -> dict:
    z, u, a, r = (np.array(c) for c in zip(*policy.labelled, strict=True))
    return {"z": z.astype(np.float32), "u": u.astype(np.float32), "a": a.astype(np.int64), "reset": r.astype(bool)}
