"""Queue only the inputs needed for corrective video pilots, with source hashes.

Uses the established private capped VM queue. No provisioning or deletion here.
Identifiers come exclusively from environment variables.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INCLUDE = ("src", "pyproject.toml", "experiments/M_release", "experiments/J_validation_transfer/configs",
           "experiments/K_recovery/configs", "experiments/L_audit_perception/configs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("command")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    bucket = os.environ["APM_GCP_BUCKET"]
    gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + a.tag
    hashes = {}
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / (job_id + ".tgz")
        with tarfile.open(archive, "w:gz") as tar:
            for item in INCLUDE:
                p = ROOT / item
                for path in sorted(p.rglob("*") if p.is_dir() else [p]):
                    if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                        continue
                    name = path.relative_to(ROOT).as_posix()
                    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    tar.add(path, arcname="apm/" + name)
            for name, data in {
                "source_snapshot.json": json.dumps({"sha256": hashes, "command": a.command}, indent=2).encode(),
                "job.sh": ("set -e\nmkdir -p out\ncp source_snapshot.json out/\n" + a.command + "\n").encode(),
            }.items():
                info = tarfile.TarInfo("apm/" + name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        if archive.stat().st_size > 30_000_000:
            raise ValueError("review inputs exceed 30 MB; inspect before uploading")
        print("archive bytes", archive.stat().st_size, flush=True)
        subprocess.run([gcloud, "storage", "cp", "-q", str(archive),
                        f"gs://{bucket}/apm-j/{a.name}/queue/{job_id}.tgz"], check=True)
    print(job_id, flush=True)


if __name__ == "__main__":
    main()
