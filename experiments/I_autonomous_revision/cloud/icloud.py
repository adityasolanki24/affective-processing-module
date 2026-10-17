"""Programme I compute: capped, self-deleting spot VMs driven over IAP SSH (no external IP, no firewall changes).

Deployment identifiers are local configuration, never repository content:
  APM_GCP_PROJECT          project ID
  APM_GCP_SERVICE_ACCOUNT  service account the VM runs as

Subcommands:
  up REGION                 temporary router + Cloud NAT in REGION, so private VMs can install packages
  create NAME ZONE          spot VM (default n2d-highcpu-128, --max-hours cap, then Google deletes it); installs the
                            Python environment (numpy only) at boot
  push NAME                 copies the working tree's code and inputs (src/, experiments/{F configs, G scripts and
                            configs, I}, runs/E2/predictor.json, pilot type models) to the VM; results live
                            in ~/results on the VM and survive pushes. Pilot, T and V runs only:
                            confirmatory runs use `push --commit` (a clean clone of HEAD), so the lock sees a
                            clean tree
  run NAME -- CMD           runs CMD in the VM's checkout (PYTHONPATH=src, venv active)
  pull NAME REMOTE LOCAL    copies a results directory back
  delete NAME               deletes the VM now
  down REGION               deletes the router and NAT (refuses while a programme-I VM exists in the region)
  audit                     lists everything of programme I that could cost money

Cost safety: spot provisioning, --max-run-duration with --instance-termination-action=DELETE (enforced by Google),
label apm-i=1 on everything, and `audit` after each session. Budget and spend: reports/COMPUTE_LOG.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

PROJECT = os.environ.get("APM_GCP_PROJECT", "").strip()
SERVICE_ACCOUNT = os.environ.get("APM_GCP_SERVICE_ACCOUNT", "").strip()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROUTER, NAT = "apm-i-router", "apm-i-nat"
INCLUDE = ("src", "experiments/F_behavioral_learning/configs", "experiments/G_architecture_selection/scripts",
           "experiments/G_architecture_selection/configs", "experiments/G_architecture_selection/seeds.py",
           "experiments/I_autonomous_revision", "runs/E2/predictor.json", "runs/I/I0/P_fit", "pyproject.toml")
STARTUP = """#!/bin/bash
set -e
apt-get update -q && apt-get install -y -q python3-venv python3-pip git
python3 -m venv /opt/venv && /opt/venv/bin/pip install -q "numpy>=2"
touch /opt/ready
"""


def g() -> str:
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        sys.exit("gcloud not found")
    return exe


def need() -> None:
    if not PROJECT or not SERVICE_ACCOUNT:
        sys.exit("set APM_GCP_PROJECT and APM_GCP_SERVICE_ACCOUNT")


def sh(args: list[str], check: bool = True, capture: bool = False) -> str:
    r = subprocess.run(args, check=check, text=True, capture_output=capture)
    return (r.stdout or "").strip() if capture else ""


def zone_of(name: str) -> str:
    z = sh([g(), "compute", "instances", "list", f"--project={PROJECT}", f"--filter=name={name}",
            "--format=value(zone.basename())"], capture=True)
    if not z:
        sys.exit(f"no VM {name}")
    return z


def ssh(name: str, command: str, check: bool = True) -> int:
    return subprocess.run([g(), "compute", "ssh", name, f"--project={PROJECT}", f"--zone={zone_of(name)}",
                           "--tunnel-through-iap", "--quiet", "--command", command], check=check).returncode


def cmd_up(a) -> None:
    need()
    sh([g(), "compute", "routers", "create", ROUTER, f"--project={PROJECT}", "--network=default",
        f"--region={a.region}"])
    sh([g(), "compute", "routers", "nats", "create", NAT, f"--router={ROUTER}", f"--project={PROJECT}",
        f"--region={a.region}", "--auto-allocate-nat-external-ips", "--nat-all-subnet-ip-ranges"])
    print(f"NAT up in {a.region}")


def cmd_down(a) -> None:
    need()
    vms = sh([g(), "compute", "instances", "list", f"--project={PROJECT}", "--filter=labels.apm-i=1",
              "--format=value(name,zone.basename())"], capture=True)
    if any(line.split()[-1].startswith(a.region) for line in vms.splitlines() if line.strip()):
        sys.exit(f"a programme-I VM still exists in {a.region}; delete it first")
    sh([g(), "compute", "routers", "nats", "delete", NAT, f"--router={ROUTER}", f"--project={PROJECT}",
        f"--region={a.region}", "--quiet"], check=False)
    for _ in range(6):
        if subprocess.run([g(), "compute", "routers", "delete", ROUTER, f"--project={PROJECT}",
                           f"--region={a.region}", "--quiet"], capture_output=True).returncode == 0:
            print(f"NAT and router deleted in {a.region}")
            return
        time.sleep(20)
    sys.exit("router could not be deleted; run audit")


def cmd_create(a) -> None:
    need()
    with tempfile.TemporaryDirectory() as tmp:
        start = Path(tmp) / "startup.sh"
        start.write_bytes(STARTUP.encode())
        sh([g(), "compute", "instances", "create", a.name, f"--project={PROJECT}", f"--zone={a.zone}",
            f"--machine-type={a.machine}", "--provisioning-model=SPOT", "--instance-termination-action=DELETE",
            f"--max-run-duration={int(a.max_hours * 3600)}s", "--image-family=ubuntu-2404-lts-amd64",
            "--image-project=ubuntu-os-cloud", "--boot-disk-size=30GB", "--no-address",
            f"--service-account={SERVICE_ACCOUNT}", "--scopes=cloud-platform", "--labels=apm-i=1",
            f"--metadata-from-file=startup-script={start}"])
    print(f"created {a.name} ({a.machine}, spot, deleted by Google after {a.max_hours} h); waiting for setup")
    for _ in range(60):
        time.sleep(15)
        if subprocess.run([g(), "compute", "ssh", a.name, f"--project={PROJECT}", f"--zone={a.zone}",
                           "--tunnel-through-iap", "--quiet", "--command", "test -f /opt/ready"],
                          capture_output=True).returncode == 0:
            print("ready")
            return
    sys.exit("setup did not finish in 15 min; check the serial console")


def _archive(path: Path, commit: bool) -> None:
    with tarfile.open(path, "w:gz") as tar:
        if commit:
            with tempfile.TemporaryDirectory() as tmp:
                clone = Path(tmp) / "repo"
                sh(["git", "clone", "-q", "--no-hardlinks", str(REPO), str(clone)])
                sh(["git", "-C", str(clone), "checkout", "-q", sh(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                                 capture=True)])
                tar.add(clone, arcname="apm")
                pred = REPO / "runs" / "E2" / "predictor.json"
                tar.add(pred, arcname="apm/runs/E2/predictor.json")
            return
        for item in INCLUDE:
            p = REPO / item
            if p.exists():
                tar.add(p, arcname=f"apm/{item}",
                        filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)


def cmd_push(a) -> None:
    need()
    with tempfile.TemporaryDirectory() as tmp:
        arc = Path(tmp) / "code.tgz"
        _archive(arc, a.commit)
        z = zone_of(a.name)
        sh([g(), "compute", "scp", str(arc), f"{a.name}:~/code.tgz", f"--project={PROJECT}", f"--zone={z}",
            "--tunnel-through-iap", "--quiet"])
    ssh(a.name, "rm -rf apm && tar xzf code.tgz && rm code.tgz && mkdir -p ~/results")
    print("pushed", "committed HEAD (clean clone)" if a.commit else "working tree")


def cmd_run(a) -> None:
    need()
    cmd = " ".join(a.cmd)
    sys.exit(ssh(a.name, f"cd ~/apm && source /opt/venv/bin/activate && export PYTHONPATH=src && {cmd}",
                 check=False))


def cmd_pull(a) -> None:
    need()
    z = zone_of(a.name)
    Path(a.local).mkdir(parents=True, exist_ok=True)
    ssh(a.name, f"cd ~/apm && tar czf ~/out.tgz {a.remote}")
    with tempfile.TemporaryDirectory() as tmp:
        sh([g(), "compute", "scp", f"{a.name}:~/out.tgz", str(Path(tmp) / "out.tgz"), f"--project={PROJECT}",
            f"--zone={z}", "--tunnel-through-iap", "--quiet"])
        with tarfile.open(Path(tmp) / "out.tgz") as tar:
            tar.extractall(a.local, filter="data")
    print(f"pulled {a.remote} -> {a.local}")


def cmd_delete(a) -> None:
    need()
    sh([g(), "compute", "instances", "delete", a.name, f"--project={PROJECT}", f"--zone={zone_of(a.name)}",
        "--quiet"])
    print("deleted", a.name)


def cmd_audit(a) -> None:
    need()
    for what, args in (("VMs", ["compute", "instances", "list", "--filter=labels.apm-i=1"]),
                       ("disks", ["compute", "disks", "list", "--filter=labels.apm-i=1"]),
                       ("routers", ["compute", "routers", "list"]),
                       ("addresses", ["compute", "addresses", "list"])):
        out = sh([g(), *args, f"--project={PROJECT}", "--format=value(name)"], capture=True)
        print(f"{what}: {out.replace(chr(10), ', ') or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    for n, f in (("up", cmd_up), ("down", cmd_down)):
        p = sub.add_parser(n)
        p.add_argument("region")
        p.set_defaults(func=f)
    p = sub.add_parser("create")
    p.add_argument("name")
    p.add_argument("zone")
    p.add_argument("--machine", default="n2d-highcpu-128")
    p.add_argument("--max-hours", type=float, default=4.0)
    p.set_defaults(func=cmd_create)
    p = sub.add_parser("push")
    p.add_argument("name")
    p.add_argument("--commit", action="store_true")
    p.set_defaults(func=cmd_push)
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("pull")
    p.add_argument("name")
    p.add_argument("remote")
    p.add_argument("local")
    p.set_defaults(func=cmd_pull)
    for n, f in (("delete", cmd_delete),):
        p = sub.add_parser(n)
        p.add_argument("name")
        p.set_defaults(func=f)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    a = ap.parse_args()
    if getattr(a, "cmd", None) and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    a.func(a)


if __name__ == "__main__":
    main()
