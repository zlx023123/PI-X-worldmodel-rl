.PHONY: install test lint doctor mock-data acceptance

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check .

doctor:
	pi0fast-wm-rl doctor

mock-data:
	pi0fast-wm-rl record --robot mock --camera mock --teleop mock --task configs/task/pick_place.yaml --episodes 10

acceptance: test doctor mock-data
	pi0fast-wm-rl inspect --dataset data/raw/pick_place
	pi0fast-wm-rl split --dataset data/raw/pick_place --train 0.8 --val 0.1 --test 0.1 --seed 42
	pi0fast-wm-rl norm-stats --dataset data/raw/pick_place
	pi0fast-wm-rl rollout --robot mock --camera mock --policy mock --episodes 3 --dry-run

