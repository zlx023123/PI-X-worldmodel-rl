#!/usr/bin/env python3
"""Convenience wrapper for deterministic Mock data collection."""

import sys

from pi0fast_wm_rl.cli import main

raise SystemExit(
    main(["record", "--robot", "mock", "--camera", "mock", "--teleop", "mock", *sys.argv[1:]])
)
