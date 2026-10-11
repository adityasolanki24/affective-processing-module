"""Programme F confirmatory runs on one capped, self-deleting Google Cloud CPU VM (pattern of NERVA's cloud/launch.py).

Deployment identifiers are local configuration, never repository content:
  APM_GCP_PROJECT          project ID (required)
  APM_GCP_BUCKET           existing bucket the VM may write to (required); everything goes under gs://BUCKET/apm-f/RUN
  APM_GCP_SERVICE_ACCOUNT  e-mail of the service account the VM runs as (needs object write on the bucket and
                           permission to delete its own instance)
  APM_GCP_ZONE             default us-central1-a

Subcommands (anything that costs money prints its plan and needs --yes):
  network-up --yes / network-down     temporary Cloud NAT so the private (no external IP) VM can install packages
  launch --machine M --max-minutes K   bundle the committed HEAD (git bundle, so the confirmatory lock sees a clean
                                       clone), upload it and start a VM that runs cloud/vm_startup.sh → cloud/job.sh
  status RUN                           VM state and the tail of the run's log
  fetch RUN                            download gs://…/apm-f/RUN/out to runs/cloud/RUN
  kill RUN                             delete the VM now
  audit                                list anything of this programme that could still cost money
  cleanup RUN --yes                    delete the run's bucket prefix (after fetching)

Cost safety: --max-run-duration with --instance-termination-action=DELETE (enforced by Google); the VM also deletes
itself when the job ends. The user-approved ceiling for programme F is US$10 in total (HANDOFF.md).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT = os.environ.get("APM_GCP_PROJECT", "").strip()
BUCKET = os.environ.get("APM_GCP_BUCKET", "").strip()
SERVICE_ACCOUNT = os.environ.get("APM_GCP_SERVICE_ACCOUNT", "").strip()
ZONE = os.environ.get("APM_GCP_ZONE", "us-central1-a").strip()
REGION = ZONE.rsplit("-", 1)[0]
ZONES = list(dict.fromkeys([ZONE] + [f"{REGION}-{z}" for z in "abcf"]))
ROUTER, NAT = "apm-f-router", "apm-f-nat"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
IMAGE_FAMILY, IMAGE_PROJECT = "ubuntu-2404-lts-amd64", "ubuntu-os-cloud"


def gcloud() -> str:
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        sys.exit("gcloud not found")
    return exe


def run(args: list[str], check: bool = True, capture: bool = False) -> str:
    print("  $ gcloud " + " ".join(args[1:]))
    r = subprocess.run(args, check=check, text=True, capture_output=capture)
    return (r.stdout or "").strip() if capture else ""


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], check=True, text=True, capture_output=True).stdout.strip()


def need_config() -> None:
    missing = [k for k, v in (("APM_GCP_PROJECT", PROJECT), ("APM_GCP_BUCKET", BUCKET),
                              ("APM_GCP_SERVICE_ACCOUNT", SERVICE_ACCOUNT)) if not v]
    if missing:
        sys.exit("set " + ", ".join(missing))


def prefix(run_id: str) -> str:
    return f"gs://{BUCKET}/apm-f/{run_id}"


def vm_name(run_id: str) -> str:
    return f"apm-f-{run_id}".lower()


def lf_copy(path: Path, directory: Path) -> Path:
    out = directory / path.name
    out.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return out


def cmd_launch(a) -> None:
    need_config()
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("uncommitted changes to tracked files: commit first (the VM runs the committed HEAD)")
    sha, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"Plan: run {run_id}, code {sha[:10]} ({branch}), {a.machine} in {ZONE}, on-demand, "
          f"hard cap {a.max_minutes} min then automatic deletion; job: cloud/{a.job}.sh as committed")
    print(f"  upload git bundle -> {prefix('RUN')}/code.bundle; VM writes {prefix('RUN')}/out and deletes itself")
    if not a.yes:
        sys.exit("not executed; re-run with --yes (this starts a billable VM)")
    g = gcloud()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "code.bundle"
        subprocess.run(["git", "-C", str(REPO), "bundle", "create", str(bundle), branch], check=True)
        run([g, "storage", "cp", str(bundle), f"{prefix(run_id)}/code.bundle"])
        startup = lf_copy(HERE / "vm_startup.sh", Path(tmp))
        base = [g, "compute", "instances", "create", vm_name(run_id), f"--project={PROJECT}",
             f"--machine-type={a.machine}", f"--image-family={IMAGE_FAMILY}", f"--image-project={IMAGE_PROJECT}",
             "--boot-disk-size=50GB", "--no-address", f"--service-account={SERVICE_ACCOUNT}", "--scopes=cloud-platform",
             f"--max-run-duration={a.max_minutes}m", "--instance-termination-action=DELETE",
             "--labels=apm-f=1", f"--metadata-from-file=startup-script={startup}",
             "--metadata=" + ",".join([f"apm-bucket={BUCKET}", f"apm-run-id={run_id}", f"apm-code-sha={sha}",
                                       f"apm-branch={branch}", f"apm-job={a.job}",
                                       f"apm-input-run={a.input_run}"])]
        for zone in ZONES:  # capacity differs by zone (stockouts)
            print(f"  trying {zone}")
            r = subprocess.run([*base, f"--zone={zone}"], text=True, capture_output=True)
            if r.returncode == 0:
                break
            print("   ", (r.stderr or "").strip().splitlines()[-1] if r.stderr else "failed")
        else:
            sys.exit("VM creation failed in every zone; nothing is running")
    print(f"started run {run_id} in {zone}; follow with: python {Path(__file__).name} status {run_id}")


def cmd_network_up(a) -> None:
    need_config()
    if not a.yes:
        sys.exit(f"not executed; re-run with --yes (creates router {ROUTER} and Cloud NAT {NAT} in {REGION})")
    g = gcloud()
    run([g, "compute", "routers", "create", ROUTER, f"--project={PROJECT}", "--network=default", f"--region={REGION}"])
    run([g, "compute", "routers", "nats", "create", NAT, f"--router={ROUTER}", f"--project={PROJECT}",
         f"--region={REGION}", "--auto-allocate-nat-external-ips", "--nat-all-subnet-ip-ranges"])


def cmd_network_down(a) -> None:
    need_config()
    g = gcloud()
    if run([g, "compute", "instances", "list", f"--project={PROJECT}", "--filter=labels.apm-f=1",
            "--format=value(name)"], capture=True):
        sys.exit("a programme-F VM still exists; delete it first")
    run([g, "compute", "routers", "nats", "delete", NAT, f"--router={ROUTER}", f"--project={PROJECT}",
         f"--region={REGION}", "--quiet"], check=False)
    for _ in range(5):  # the router can briefly stay "in use" right after the NAT is deleted
        if subprocess.run([g, "compute", "routers", "delete", ROUTER, f"--project={PROJECT}", f"--region={REGION}",
                           "--quiet"], capture_output=True).returncode == 0:
            print(f"deleted {NAT} and {ROUTER}")
            return
        time.sleep(30)
    sys.exit(f"router {ROUTER} could not be deleted; check with: audit")


def cmd_status(a) -> None:
    need_config()
    g = gcloud()
    print(run([g, "compute", "instances", "list", f"--project={PROJECT}", "--filter=labels.apm-f=1",
               "--format=table(name,zone.basename(),machineType.basename(),status,creationTimestamp)"],
              capture=True) or "no programme-F VM")
    log = subprocess.run([g, "storage", "cat", f"{prefix(a.run)}/out/job.log"], text=True, capture_output=True)
    print("\n".join((log.stdout or "(no log yet)").splitlines()[-a.lines:]))


def cmd_fetch(a) -> None:
    need_config()
    dest = REPO / "runs" / "cloud" / a.run
    dest.mkdir(parents=True, exist_ok=True)
    run([gcloud(), "storage", "rsync", "--recursive", f"{prefix(a.run)}/out", str(dest)])
    print(f"fetched to {dest}")


def cmd_kill(a) -> None:
    need_config()
    g = gcloud()
    zone = run([g, "compute", "instances", "list", f"--project={PROJECT}", f"--filter=name={vm_name(a.run)}",
                "--format=value(zone.basename())"], capture=True)
    if zone:
        run([g, "compute", "instances", "delete", vm_name(a.run), f"--project={PROJECT}", f"--zone={zone}", "--quiet"])
    else:
        print("no such VM")


def cmd_audit(a) -> None:
    need_config()
    g = gcloud()
    for what, args in (("VMs", ["compute", "instances", "list", "--filter=labels.apm-f=1"]),
                       ("disks", ["compute", "disks", "list", "--filter=name~^apm-f-"]),
                       ("addresses", ["compute", "addresses", "list"]),
                       ("routers", ["compute", "routers", "list"])):
        print(f"{what}:", run([g, *args, f"--project={PROJECT}", "--format=value(name)"], capture=True) or "none")
    objs = subprocess.run([g, "storage", "ls", f"gs://{BUCKET}/apm-f/"], text=True, capture_output=True)
    print("bucket objects under apm-f/:", (objs.stdout or "").strip() or "none")


def cmd_cleanup(a) -> None:
    need_config()
    if not a.yes:
        sys.exit("not executed; re-run with --yes (deletes the run's bucket prefix; fetch first)")
    run([gcloud(), "storage", "rm", "--recursive", prefix(a.run)], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--machine", default="n2d-highcpu-96")
    p.add_argument("--max-minutes", type=int, required=True)
    p.add_argument("--job", required=True, choices=sorted(f.stem for f in HERE.glob("*.sh") if f.stem != "vm_startup"))
    p.add_argument("--input-run", default="", help="earlier run id whose outputs the job reads")
    p.add_argument("--yes", action="store_true")
    for name in ("status", "fetch", "kill", "cleanup"):
        q = sub.add_parser(name)
        q.add_argument("run")
        q.add_argument("--lines", type=int, default=30)
        q.add_argument("--yes", action="store_true")
    sub.add_parser("audit")
    sub.add_parser("network-up").add_argument("--yes", action="store_true")
    sub.add_parser("network-down")
    a = ap.parse_args()
    {"launch": cmd_launch, "status": cmd_status, "fetch": cmd_fetch, "kill": cmd_kill, "audit": cmd_audit,
     "cleanup": cmd_cleanup, "network-up": cmd_network_up, "network-down": cmd_network_down}[a.cmd](a)


if __name__ == "__main__":
    main()
