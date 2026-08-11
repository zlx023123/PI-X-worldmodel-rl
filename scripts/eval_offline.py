#!/usr/bin/env python3
"""Convenience wrapper for offline evaluation."""

import sys

from pi0fast_wm_rl.cli import main

raise SystemExit(main(["eval-offline", *sys.argv[1:]]))
