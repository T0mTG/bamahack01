"""Reinforcement learning for the longitudinal (throttle/brake) policy.

Training code only. The submission is ``controllers/rl_longitudinal/`` and never imports this
package; this package imports the controller's modules so training and driving share one source
of truth for planning, features and the safety shield.
"""
import sys
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parent.parent / "controllers" / "rl_longitudinal"
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))
