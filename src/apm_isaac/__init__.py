"""APM v1.0 Isaac Sim integration (needs the pinned Isaac Sim 5.1 environment; see experiments/M_release)."""

# Kit command-line arguments for scripts with animated people: the AnimationGraph schema must be registered at
# startup, e.g. AppLauncher({"headless": True, "enable_cameras": True, "kit_args": ANIM_KIT_ARGS}).
ANIM_KIT_ARGS = " ".join(f"--enable {e}" for e in ("omni.anim.graph.schema", "omni.anim.graph.core",
                                                   "omni.anim.retarget.core", "omni.anim.navigation.schema"))


def hard_exit(code: int = 0) -> None:
    """End an Isaac script after its results are written. `app.close()` can hang with cameras enabled, which
    stalls chained jobs; this flushes and exits the process directly."""
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def exit_on_error() -> None:
    """Make an uncaught exception end the process. Kit keeps a standalone script's process alive after a
    Python error, which turns a crash into a silent hang until the job's timeout (M6 calibration runs)."""
    import sys
    import traceback

    def hook(etype, value, tb):
        traceback.print_exception(etype, value, tb)
        hard_exit(1)
    sys.excepthook = hook
