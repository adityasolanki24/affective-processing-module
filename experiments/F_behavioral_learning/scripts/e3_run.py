"""E3: experience, changing relationships and false attribution (MASTER_PREREGISTRATION.md §2 E3).

One job = (scenario, partition, lifetime index, system). Every system lives through the same 30-episode lifetime
(common random numbers). Evaluation-only observers ride along on the identical observation stream and never
influence a decision:
  * a recording copy of the explicit memory (`ObsMemory`, the same deterministic class B1/B2 use, so its state equals
    the policy's own memory; checked at the end of each lifetime) that logs, per encounter, the memory's prediction
    at encounter start and whether a collision with that entity followed;
  * the naive temporal-association memory (credits any adverse outcome, collision or slip, within 1 s to every
    entity within 1.5 m), an evaluation reference only.
Ground truth (true identity and disposition of each track) is read from the simulator by the observer only.

    python scripts/e3_run.py --partition P --n 20
    python scripts/e3_run.py --partition X --n <addendum>        # requires the committed E3 addendum
    python scripts/e3_run.py --partition P --n 10 --mode retention
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import seeds as S  # noqa: E402
from e2_common import _STATE, _init, make_policy  # noqa: E402

from apm.memory.entities import BASE_RATE, PRIOR_STRENGTH  # noqa: E402
from apm_research.harness import EPISODE_GAP_S, Lifetime, run_episode  # noqa: E402
from apm_research.memory import ENTITY_TTL_S, ObsMemory  # noqa: E402
from apm_research.policies import B1Eps  # noqa: E402
from apm_research.provenance import confirmatory_lock, environment, write_json  # noqa: E402
from apm_testbeds.wheeled_ctrl import GO, Identity, WorldConfig, child  # noqa: E402

EPISODES, SWITCH = 30, 15
NEAR_M, NAIVE_RADIUS_M, NAIVE_WINDOW_S = 1.5, 1.5, 1.0
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
SYSTEMS = {"S1": ("B0", "B1", "B1eps", "B2", "Oracle"), "S6": ("B0", "B1", "B1eps", "B2", "Oracle")}
DEFAULT_SYSTEMS = ("B0", "B1", "B2", "Oracle")


def with_dispositions(base: list[Identity], fixed: dict[str, str]) -> list[Identity]:
    return [Identity(i.name, fixed.get(i.name, i.disposition)) for i in base]


def make_lifetime(scenario: str, ss: np.random.SeedSequence) -> tuple[Lifetime, dict]:
    """Scenario family → lifetime (schedule of dispositions in force per episode) and evaluation roles."""
    cfg, roles = WorldConfig(), {}
    if scenario == "S1":   # two identities hazardous → benign at episode 15
        def sched(e, base):
            return with_dispositions(base, {"ped-0": "careless" if e < SWITCH else "careful",
                                            "ped-1": "careless" if e < SWITCH else "careful"})
        roles = {"switched": ["ped-0", "ped-1"], "to": "careful"}
    elif scenario == "S2":  # benign → hazardous
        def sched(e, base):
            return with_dispositions(base, {"ped-0": "careful" if e < SWITCH else "careless",
                                            "ped-1": "careful" if e < SWITCH else "careless"})
        roles = {"switched": ["ped-0", "ped-1"], "to": "careless"}
    elif scenario in ("S3", "S4"):  # stable mixed histories (S4: with identity errors)
        def sched(e, base):
            return with_dispositions(base, {"ped-0": "careless", "ped-1": "careful"})
        if scenario == "S4":
            cfg = WorldConfig(identity_error=0.2, identity_ambiguous=0.1)
    elif scenario == "S5":  # a careful identity frequents low-friction patches: slips co-occur with it
        cfg = WorldConfig(patches=(1, 3), patch_affinity=("ped-0",), force_present=("ped-0",))
        def sched(e, base):
            return with_dispositions(base, {"ped-0": "careful", "ped-1": "careful"})
        roles = {"affine": "ped-0"}
    elif scenario == "S6":  # censoring: a careless identity the policy learns to avoid
        cfg = WorldConfig(force_present=("ped-0",))
        def sched(e, base):
            return with_dispositions(base, {"ped-0": "careless"})
        roles = {"avoided": "ped-0"}
    elif scenario == "S7":  # same-class newcomer after a careless-heavy history
        def sched(e, base):
            pop = with_dispositions(base, {"ped-0": "careless", "ped-1": "careless", "ped-2": "careless",
                                           "ped-3": "careless", "ped-4": "careful", "ped-5": "careful"})
            return pop if e >= SWITCH else [i for i in pop if i.name != "ped-5"]
        roles = {"newcomer": "ped-5"}
    else:
        raise ValueError(scenario)
    return Lifetime(ss, n_episodes=EPISODES, config=cfg, schedule=sched), roles


class RecordingMemory(ObsMemory):
    """ObsMemory plus an encounter log with ground truth (observer only)."""

    def __init__(self) -> None:
        super().__init__()
        self.open: dict[str, list] = {}
        self.log: list[list] = []
        self.truth: dict[str, tuple[str | None, str | None]] = {}
        self.episode = 0

    def observe(self, obs, t: float) -> None:
        for e in obs.entities:
            if e.kind == "person" and e.track not in self.mem.encounters:
                p, conf = self.p_collision(e.identity, "person")
                name, disp = self.truth.get(e.track, (None, None))
                self.open[e.track] = [self.episode, name, disp, e.identity, round(p, 5), round(conf, 4), 0]
        for ev in obs.events:
            if ev[0] == "collision" and ev[1] in self.open:
                self.open[ev[1]][6] = 1
        super().observe(obs, t)

    def expire(self, t: float) -> None:
        for track in [k for k, s in self.last_seen.items() if t - s > ENTITY_TTL_S]:
            if track in self.open:
                self.log.append(self.open.pop(track))
        super().expire(t)


class NaiveMemory:
    """Temporal-association reference: any adverse outcome credits every entity within 1.5 m in the last 1 s."""

    def __init__(self) -> None:
        self.n: dict[str, float] = {}
        self.k: dict[str, float] = {}
        self.open: dict[str, list] = {}  # track → [identity, adverse]
        self.last_seen: dict[str, float] = {}
        self.last_near: dict[str, float] = {}

    def observe(self, obs, t: float) -> None:
        for e in obs.entities:
            if e.kind != "person":
                continue
            if e.track not in self.open:
                self.open[e.track] = [e.identity, 0]
            self.last_seen[e.track] = t
            if e.distance <= NAIVE_RADIUS_M:
                self.last_near[e.track] = t
        if any(ev[0] in ("collision", "slip") for ev in obs.events):
            for track, s in self.last_near.items():
                if t - s <= NAIVE_WINDOW_S and track in self.open:
                    self.open[track][1] = 1
        self.expire(t)

    def expire(self, t: float) -> None:
        for track in [k for k, s in self.last_seen.items() if t - s > ENTITY_TTL_S]:
            ident, adverse = self.open.pop(track)
            del self.last_seen[track]
            self.last_near.pop(track, None)
            if ident is not None:
                self.n[ident] = self.n.get(ident, 0.0) + 1
                self.k[ident] = self.k.get(ident, 0.0) + adverse

    def p(self, ident: str) -> float:
        return (self.k.get(ident, 0.0) + PRIOR_STRENGTH * BASE_RATE) / (self.n.get(ident, 0.0) + PRIOR_STRENGTH)


class Probe:
    """Forwards everything to the policy unchanged; feeds the observers; logs decisions near people with truth."""

    def __init__(self, policy, world, rec: RecordingMemory, naive: NaiveMemory, episode: int):
        self.policy, self.world, self.rec, self.naive, self.episode = policy, world, rec, naive, episode
        self.near: list[tuple[int, str | None, int]] = []
        self.last_obs = None

    def _truth(self) -> None:
        for oid, (_, _, name, disp, _) in self.world.truth()["people"].items():
            self.rec.truth[oid] = (name, disp)

    def begin_episode(self, t0: float) -> None:
        self._truth()
        self.rec.episode = self.episode
        self.rec.expire(t0)
        self.naive.expire(t0)
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        self.last_obs = obs
        self.rec.observe(obs, t)
        self.naive.observe(obs, t)
        self.policy.observe(obs, t)

    def decide(self, t: float) -> int:
        a = self.policy.decide(t)
        obs = self.last_obs
        people = [e for e in obs.entities if e.kind == "person"] if obs is not None else []
        if people:
            nearest = min(people, key=lambda e: e.distance)
            if nearest.distance < NEAR_M:
                self.near.append((self.episode, self.rec.truth.get(nearest.track, (None, None))[1], int(a)))
        return a

    def end_episode(self, t: float) -> None:
        self.policy.end_episode(t)


def build_policy(spec: dict, ss: np.random.SeedSequence):
    if spec["system"] == "B1eps":
        return B1Eps(_STATE["predictor"], spec["lam_c"], spec["lam_s"], eps=0.1, radius=NEAR_M, seed=child(ss, 600))
    return make_policy(spec)


def evidence(memory: ObsMemory) -> dict:
    return {k: (round(r["encounters"], 6), {kk: round(v, 6) for kk, v in r["counts"].items()})
            for k, r in memory.mem.to_dict()["records"].items()}


def run_job(job: dict) -> dict:
    ss = S.world_seed("E3", job["partition"], job["index"])
    lt, roles = make_lifetime(job["scenario"], ss)
    spec = job["spec"]
    policy = build_policy(spec, ss)
    policy.begin_lifetime()
    rec, naive = RecordingMemory(), NaiveMemory()
    base = lt.population()
    names = [i.name for i in base]
    rows, snaps, near = [], [], []
    for e in range(EPISODES):
        pop = lt.schedule(e, base) if lt.schedule else base
        truth = {i.name: i.disposition for i in pop}
        if spec.get("oracle_by_disposition"):
            policy.oracle_rates = {n: spec["oracle_by_disposition"][d] for n, d in truth.items()}
        world = lt.world(e)
        probe = Probe(policy, world, rec, naive, e)
        t0 = e * (lt.config.episode_s + EPISODE_GAP_S)
        row = run_episode(world, probe, t0)
        row.pop("collision_log")
        row["episode"] = e
        rows.append(row)
        near += probe.near
        rec.expire(t0 + lt.config.episode_s + ENTITY_TTL_S + 0.01)  # close this episode's encounters for the snapshot
        naive.expire(t0 + lt.config.episode_s + ENTITY_TTL_S + 0.01)
        for n in names:
            p, conf = rec.p_collision(n, "person")
            snaps.append((e, n, truth.get(n), round(p, 5), round(conf, 4), round(naive.p(n), 5)))
    rec.expire(1e12)
    same = None
    if hasattr(policy, "memory"):
        policy.memory.expire(1e12)
        same = evidence(policy.memory) == evidence(rec)  # encounter counts and outcome evidence (not timestamps)
    return {"scenario": job["scenario"], "partition": job["partition"], "index": job["index"],
            "system": spec["label"], "roles": roles, "rows": rows, "snapshots": snaps, "encounters": rec.log,
            "near": near, "observer_equals_policy_memory": same,
            "population": {i.name: i.disposition for i in base}}


# ---------------------------------------------------------------------------------------------- retention / history
def recorded_stream(lt: Lifetime, episode: int) -> list:
    world = lt.world(episode)
    stream = [world.observe_now()]
    while not world.done:
        stream.append(world.step(GO))
    return stream


def replay(policy, stream: list, t0: float) -> list[int]:
    """Open-loop decisions of a policy on a recorded observation stream (one per decision period)."""
    policy.begin_episode(t0)
    out = []
    for i, obs in enumerate(stream):
        policy.observe(obs, t0 + obs.t)
        if i % 10 == 0:
            out.append(int(policy.decide(t0 + obs.t)))
    return out


def retention_job(job: dict) -> dict:
    """Save/kill/restore at the episode-15 boundary (B1 and B2) and history dependence (B1).

    Retention: a policy lives episodes 0–14; its state is saved (B1: `EntityMemory.to_dict`; B2: plus `APM.save`),
    the process state is discarded and a fresh policy is restored from the saved state. Both the uninterrupted and
    the restored policy then decide open loop on one identical recorded observation stream (episode 15 under B0);
    B1 is also compared closed loop over episodes 15–29.
    History dependence: B1 after this lifetime's history vs B1 after another lifetime's history, open loop on the
    same recorded stream."""
    from apm.memory.entities import EntityMemory
    from apm_research.policies import AblatedAPM

    ss = S.world_seed("E3", job["partition"], job["index"])
    lt, _ = make_lifetime("S3", ss)
    period = lt.config.episode_s + EPISODE_GAP_S
    stream = recorded_stream(lt, SWITCH)
    tmp = Path(job["tmp"]) / f"apm_{job['index']}.json"
    out: dict = {"index": job["index"]}

    def live(spec, life, episodes):
        policy = make_policy(spec)
        policy.begin_lifetime()
        for e in range(episodes):
            run_episode(life.world(e), policy, e * period)
        return policy

    for label in ("B1", "B2"):
        spec = job["specs"][label]
        a = live(spec, lt, SWITCH)
        a.memory.expire(SWITCH * period)
        saved = json.loads(json.dumps(a.memory.mem.to_dict()))
        if label == "B2":
            a.apm.save(tmp)
        b = make_policy(spec)
        b.begin_lifetime()
        b.memory.mem = EntityMemory.from_dict(saved)
        if label == "B2":
            b.apm = AblatedAPM.load(tmp, tonic=b.tonic, memory_frames=b.memory_frames)
        memory_identical = a.memory.mem.to_dict() == b.memory.mem.to_dict()
        da, db = replay(copy.deepcopy(a), stream, SWITCH * period), replay(copy.deepcopy(b), stream, SWITCH * period)
        row = {"memory_identical": memory_identical, "replay_decision_agreement": float(np.mean(np.array(da) ==
                                                                                                np.array(db)))}
        if label == "B1":
            acts = []
            for p in (a, b):
                acts.append([run_episode(lt.world(e), p, e * period)["actions"] for e in range(SWITCH, EPISODES)])
            row["closed_loop_actions_identical"] = acts[0] == acts[1]
        out[label] = row
    other, _ = make_lifetime("S3", S.world_seed("E3", job["partition"], job["index"] + 10_000))
    h0, h1 = live(job["specs"]["B1"], lt, SWITCH), live(job["specs"]["B1"], other, SWITCH)
    d0, d1 = replay(h0, stream, SWITCH * period), replay(h1, stream, SWITCH * period)
    out["history_divergence"] = float(np.mean(np.array(d0) != np.array(d1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--tuned", type=Path, default=HERE / "configs" / "E2_tuned.json")
    ap.add_argument("--out", type=Path, default=Path("runs/E3"))
    ap.add_argument("--mode", choices=("main", "retention"), default="main")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    provenance = (confirmatory_lock(HERE / "prereg" / "E3_ADDENDUM.md") if args.partition in ("X", "S")
                  else environment())
    start = time.time()
    tuned = json.loads(args.tuned.read_text())["best"]
    b1 = {"lam_c": tuned["B1"]["lam_c"], "lam_s": tuned["B1"]["lam_s"]}
    b2 = {k: tuned["B2"][k] for k in ("lam_c", "lam_s", "k_pad", "k_t")}
    rates = json.loads((HERE / "configs" / "oracle_rates.json").read_text())["rates_by_disposition"]
    specs = {"B0": {"system": "B0"}, "B1": {"system": "B1", **b1}, "B1eps": {"system": "B1eps", **b1},
             "B2": {"system": "B2", **b2}, "Oracle": {"system": "B1", **b1, "oracle_by_disposition": rates}}
    for k, v in specs.items():
        v["label"] = k
    out_dir = args.out / args.partition
    out_dir.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    if args.mode == "retention":
        jobs = [{"partition": args.partition, "index": args.first + i, "specs": specs, "tmp": str(out_dir)}
                for i in range(args.n)]
        with Pool(args.workers, initializer=_init, initargs=(args.predictor,)) as pool:
            results = pool.map(retention_job, jobs, chunksize=1)
        write_json(out_dir / "retention.json", {"provenance": provenance, "results": results,
                                                "wall_s": round(time.time() - start, 1)})
        print(json.dumps(results[:3]))
        return
    jobs = [{"scenario": sc, "partition": args.partition, "index": args.first + i, "spec": copy.deepcopy(specs[s])}
            for sc in args.scenarios.split(",") for s in SYSTEMS.get(sc, DEFAULT_SYSTEMS) for i in range(args.n)]
    jobs.sort(key=lambda j: j["spec"]["system"] != "B2")  # slowest first
    with Pool(args.workers, initializer=_init, initargs=(args.predictor,)) as pool, \
            open(out_dir / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            fh.write(json.dumps(r, default=float) + "\n")
    write_json(out_dir / "run_meta.json", {"provenance": provenance, "specs": specs, "n": args.n,
                                           "first": args.first, "episodes": EPISODES, "jobs": len(jobs),
                                           "wall_s": round(time.time() - start, 1)})
    print("done", len(jobs), round(time.time() - start, 1))


if __name__ == "__main__":
    main()
