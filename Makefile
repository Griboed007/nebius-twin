.PHONY: install test synth job endpoint preflight push teardown

# Interpreter: override with `make PYTHON=python synth` if `python` is your venv.
PYTHON ?= python3

install:
	pip install -r requirements.txt
test:
	pytest -q
synth:
	$(PYTHON) -m src.data.synthetic --out artifacts/synthetic.parquet --seed 7
job:
	$(PYTHON) -m services.batch_job.run
endpoint:
	uvicorn services.twin_api.app:app --host 0.0.0.0 --port 8080
preflight:
	bash deploy/preflight.sh
push:
	bash deploy/build_push.sh
teardown:
	bash deploy/teardown.sh
