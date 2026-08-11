#!/usr/bin/env python3
"""Convenience wrapper for normalization statistics."""

import sys

from pi0fast_wm_rl.cli import main

raise SystemExit(main(["norm-stats", *sys.argv[1:]]))
