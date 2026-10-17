"""Programme I compute: capped, self-deleting spot VMs that run jobs from a bucket queue (`runner.sh`).

No SSH: the project's organization enforces OS Login. Jobs and results travel through the bucket the VM's service
account can write to.

Deployment identifiers are local configuration, never repository content:
  APM_GCP_PROJECT          project ID
  APM_GCP_BUCKET           bucket for the queue and results (gs://BUCKET/apm-i/NAME/...)
  APM_GCP_SERVICE_ACCOUNT  service account the VM runs as (bucket write; may delete its own instance)

Subcommands:
  up REGION                 temporary router + Cloud NAT in REGION, so private VMs can install packages
  down REGION               deletes them (refuses while a programme-I VM exists in the region)
  create NAME ZONE          spot VM (default n2d-highcpu-128). Google deletes it after --max-hours; it deletes
                            itself after --idle-min minutes without a job. Waits until the runner is ready
  submit NAME CMD [--commit] [--tag T]
                            queues CMD, run in a fresh copy of the code with PYTHONPATH=src. Outputs must go to
                            out/. Default: the working tree's code and inputs (pilot/T/V). --commit: a clean
                            clone of HEAD with .git, which confirmatory runs need for the lock. Prints the job id
  wait NAME ID              blocks until the job is done; prints exit code and log tail
  fetch NAME ID DEST        downloads and unpacks out/ into DEST
  delete NAME               deletes the VM now
  audit                     lists programme-I VMs, disks, routers and addresses
  cleanup NAME              deletes gs://BUCKET/apm-i/NAME (after fetching)

Spend: reports/COMPUTE_LOG.md. User cap for programme I: US$40.
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
BUCKET = os.environ.get("APM_GCP_BUCKET", "").strip()
SERVICE_ACCOUNT = os.environ.get("APM_GCP_SERVICE_ACCOUNT", "").strip()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROUTER, NAT = "apm-i-router", "apm-i-nat"
INCLUDE = ("src", "pyproject.toml", "experiments/F_behavioral_learning/configs",
           "experiments/G_architecture_selection/scripts", "experiments/G_architecture_selection/configs",
           "experiments/G_architecture_selection/seeds.py", "experiments/I_autonomous_revision",
           "runs/E2/predictor.json", "runs/I/I0/P_fit")


def g() -> str:
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        sys.exit("gcloud not found")
    return exe


def need() -> None:
    if not (PROJECT and BUCKET and SERVICE_ACCOUNT):
        sys.exit("set APM_GCP_PROJECT, APM_GCP_BUCKET and APM_GCP_SERVICE_ACCOUNT")


def sh(args: list[str], check: bool = True, capture: bool = False) -> str:
    r = subprocess.run(args, check=check, text=True, capture_output=capture)
    return (r.stdout or "").strip() if capture else ""


def q(name: str) -> str:
    return f"gs://{BUCKET}/apm-i/{name}"


def exists(uri: str) -> bool:
    return subprocess.run([g(), "storage", "ls", uri], capture_output=True).returncode == 0


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
        start = Path(tmp) / "runner.sh"
        start.write_bytes((HERE / "runner.sh").read_bytes().replace(b"\r\n", b"\n"))
        sh([g(), "compute", "instances", "create", a.name, f"--project={PROJECT}", f"--zone={a.zone}",
            f"--machine-type={a.machine}", "--provisioning-model=SPOT", "--instance-termination-action=DELETE",
            f"--max-run-duration={int(a.max_hours * 3600)}s", "--image-family=ubuntu-2404-lts-amd64",
            "--image-project=ubuntu-os-cloud", "--boot-disk-size=30GB", "--no-address",
            f"--service-account={SERVICE_ACCOUNT}", "--scopes=cloud-platform", "--labels=apm-i=1",
            f"--metadata-from-file=startup-script={start}",
            f"--metadata=apm-bucket={BUCKET},apm-idle-min={a.idle_min}"])
    print(f"created {a.name} ({a.machine}, spot; hard cap {a.max_hours} h, idle delete {a.idle_min} min)")
    for _ in range(80):
        if exists(f"{q(a.name)}/ready"):
            print("ready")
            return
        time.sleep(15)
    sys.exit("runner not ready after 20 min")


def _archive(path: Path, commit: bool, cmd: str) -> str:
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(path, "w:gz") as tar:
        job = Path(tmp) / "job.sh"
        job.write_bytes(f"set -e\nmkdir -p out\n{cmd}\n".encode())
        if commit:
            clone = Path(tmp) / "repo"
            sh(["git", "clone", "-q", "--no-hardlinks", str(REPO), str(clone)])
            sha = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture=True)
            sh(["git", "-C", str(clone), "checkout", "-q", sha])
            tar.add(clone, arcname="apm")
            for extra in ("runs/E2/predictor.json",):  # git-tracked inputs are already in the clone
                if not (clone / extra).exists():
                    tar.add(REPO / extra, arcname=f"apm/{extra}")
            label = f"commit {sha[:10]}"
        else:
            for item in INCLUDE:
                p = REPO / item
                if p.exists():
                    tar.add(p, arcname=f"apm/{item}",
                            filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
            label = "working tree"
        tar.add(job, arcname="apm/job.sh")
    return label


def cmd_submit(a) -> None:
    need()
    job_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{a.tag}" if a.tag else "")
    with tempfile.TemporaryDirectory() as tmp:
        arc = Path(tmp) / f"{job_id}.tgz"
        label = _archive(arc, a.commit, a.cmd)
        sh([g(), "storage", "cp", "-q", str(arc), f"{q(a.name)}/queue/{job_id}.tgz"])
    print(job_id, f"({label})")


def cmd_wait(a) -> None:
    need()
    while not exists(f"{q(a.name)}/done/{a.id}/rc"):
        time.sleep(15)
    rc = sh([g(), "storage", "cat", f"{q(a.name)}/done/{a.id}/rc"], capture=True)
    log = sh([g(), "storage", "cat", f"{q(a.name)}/done/{a.id}/log.txt"], capture=True)
    print("\n".join(log.splitlines()[-a.lines:]))
    print("exit code", rc)


def cmd_fetch(a) -> None:
    need()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        sh([g(), "storage", "cp", "-q", f"{q(a.name)}/done/{a.id}/out.tgz", f"{tmp}/out.tgz"])
        with tarfile.open(f"{tmp}/out.tgz") as tar:
            tar.extractall(dest, filter="data")
    print(f"fetched {a.id} -> {dest}")


def cmd_delete(a) -> None:
    need()
    z = sh([g(), "compute", "instances", "list", f"--project={PROJECT}", f"--filter=name={a.name}",
            "--format=value(zone.basename())"], capture=True)
    if not z:
        print("no such VM")
        return
    sh([g(), "compute", "instances", "delete", a.name, f"--project={PROJECT}", f"--zone={z}", "--quiet"])
    print("deleted", a.name)


def cmd_audit(a) -> None:
    need()
    for what, args in (("VMs", ["compute", "instances", "list", "--filter=labels.apm-i=1"]),
                       ("disks", ["compute", "disks", "list", "--filter=labels.apm-i=1"]),
                       ("routers", ["compute", "routers", "list"]),
                       ("addresses", ["compute", "addresses", "list"])):
        out = sh([g(), *args, f"--project={PROJECT}", "--format=value(name)"], capture=True)
        print(f"{what}: {out.replace(chr(10), ', ') or 'none'}")


def cmd_cleanup(a) -> None:
    need()
    sh([g(), "storage", "rm", "--recursive", "-q", q(a.name)], check=False)
    print("removed", q(a.name))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    for n, f in (("up", cmd_up), ("down", cmd_down)):
        p = sub.add_parser(n)
        p.add_argument("region")
        p.set_defaults(func=f)
    p = sub.add_parser("create")
    p.add_argument("name")
    p.add_argument("zone")
    p.add_argument("--machine", default="n2d-highcpu-128")
    p.add_argument("--max-hours", type=float, default=4.0)
    p.add_argument("--idle-min", type=int, default=30)
    p.set_defaults(func=cmd_create)
    p = sub.add_parser("submit")
    p.add_argument("name")
    p.add_argument("cmd")
    p.add_argument("--commit", action="store_true")
    p.add_argument("--tag", default="")
    p.set_defaults(func=cmd_submit)
    p = sub.add_parser("wait")
    p.add_argument("name")
    p.add_argument("id")
    p.add_argument("--lines", type=int, default=5)
    p.set_defaults(func=cmd_wait)
    p = sub.add_parser("fetch")
    p.add_argument("name")
    p.add_argument("id")
    p.add_argument("dest")
    p.set_defaults(func=cmd_fetch)
    for n, f in (("delete", cmd_delete), ("cleanup", cmd_cleanup)):
        p = sub.add_parser(n)
        p.add_argument("name")
        p.set_defaults(func=f)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
