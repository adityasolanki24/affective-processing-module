import json
import sys
import time

sys.path.insert(0, "src")
import numpy as np

from apm_transfer.expression_map import GRID
from apm_transfer.vision import VisionCamera

robot = sys.argv[1]
if robot == "wheeled":
    from apm_transfer.wheeled_world import closed_loop_encounter as enc
    kw = {}
elif robot == "humanoid":
    from apm_transfer.humanoid_world import closed_loop_encounter as enc
    kw = {}
else:
    from apm_transfer.nerva_world import closed_loop_encounter as enc
    cfg = "experiments/J_validation_transfer/configs/H_admitted_interface.json"
    kw = {"channels": json.load(open(cfg))["interface"]}
cam = VisionCamera(keep_frames=True)
for seed, et in ((1, "pushy"), (2, "calm"), (3, "pushy")):
    t0 = time.time()
    ev, out, truth = enc(seed, et, GRID[2], windows=3, camera=cam, abort_on_detect=False, **kw)
    v = out["vision"]
    print(robot, et, "t", round(time.time() - t0, 1), "codes", out["camera"], "est", out["camera_reaches"],
          "truth", [[(t["reach"], t["in_view"], t["dist"]) for t in w] for w in v["truth"]],
          "frames", len(v["frames"]), "d", out["final_distance"])
np.save(f"runs/L/pilot/vis_frames_{robot}.npy", np.array(v["frames"], np.float32))
