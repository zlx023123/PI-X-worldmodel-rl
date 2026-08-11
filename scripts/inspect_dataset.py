#!/usr/bin/env python3
"""Convenience wrapper for dataset validation."""

import sys

from pi0fast_wm_rl.cli import main

raise SystemExit(main(["inspect", *sys.argv[1:]]))
