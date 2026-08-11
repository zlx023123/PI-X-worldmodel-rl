#!/usr/bin/env bash
set -euo pipefail

pi0fast-wm-rl train-act --config configs/policy/act.yaml "$@"

