#!/usr/bin/env python3
"""Convenience wrapper for episode-level splitting."""

import sys

from pi0fast_wm_rl.cli import main

raise SystemExit(main(["split", *sys.argv[1:]]))
