#!/usr/bin/env python3
"""Convenience wrapper for safety-filtered Mock rollout."""

import sys

from pi0fast_wm_rl.cli import main

raise SystemExit(
    main(["rollout", "--robot", "mock", "--camera", "mock", "--policy", "mock", *sys.argv[1:]])
)
