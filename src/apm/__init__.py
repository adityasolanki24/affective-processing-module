"""APM: Affective Processing Module, a portable, experience-grounded affective regulation and learning module.

The core depends only on NumPy. Robots connect through AffectLink (apm.protocol) via embodiment adapters
(apm.adapters); the learning stack (apm.learning) is optional and replaceable without changing AffectLink.
"""
from apm.core.runtime import APM, Outputs

__version__ = "0.1.0"
__all__ = ["APM", "Outputs", "__version__"]
