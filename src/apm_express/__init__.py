"""Programme H (APM v0.4): a removable expressive layer on top of the frozen APM v0.3 architecture.

    ExpressionIntent -> EmbodimentExpressionAdapter -> ValidatedMotionController   (+ independent SafetyMonitor)

Nothing here modifies `apm`, `apm_research`, `apm_select` or `apm_testbeds`, and nothing here can emit joint
targets: learned or scripted components produce an `ExpressionIntent`, adapters turn it into requests on validated
channels only. `biped` (NERVA) imports MuJoCo lazily; the rest is numpy-only.
"""
