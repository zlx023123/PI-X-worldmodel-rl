#!/usr/bin/env bash
set -euo pipefail

pi0fast-wm-rl train-pi0fast --config configs/policy/pi0fast_bc.yaml "$@"

