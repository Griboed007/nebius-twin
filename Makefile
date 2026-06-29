.PHONY: install test synth job endpoint teardown
install:
	pip install -r requirements.txt
test:
	pytest -q
synth:
	python -m src.data.synthetic --out artifacts/synthetic.parquet --seed 7
job:
	python -m services.batch_job.run
endpoint:
	uvicorn services.twin_api.app:app --host 0.0.0.0 --port 8080
teardown:
	bash deploy/teardown.sh
