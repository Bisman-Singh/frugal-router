.PHONY: install test eval docker demo

install:
	uv pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

eval:
	.venv/bin/frugal eval --dataset data/dev_tasks.jsonl --out runs/latest

docker:
	docker build -t frugal-router .

model:
	bash scripts/download_model.sh
