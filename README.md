# Semantic Digital Twin for Vehicle Telematics on Nebius Serverless

A reproducible Physical-AI pipeline: a Nebius Serverless **Job** enriches GPS
trajectories with hybrid trajectory prediction + anomaly scoring and writes a
**SAREF/RDF** twin; a Nebius Serverless **Endpoint** serves live predictions and
**SPARQL** queries over it. Built for the Nebius Serverless AI Builders Challenge.

## Reproduce offline (no downloads)
```
pip install -r requirements.txt
make synth && make test && make job
```
Produces `enriched.parquet`, `twin.ttl`, `metrics.json` in `artifacts/out/`.

## Run on Nebius
See `deploy/`. Set `.env` from `.env.example`, then `deploy/job.sh` and
`deploy/endpoint.sh`; tear down with `make teardown`. Proof in `docs/proof/`.

## Architecture
Job → object storage (SAREF/RDF twin) → Endpoint, sharing one tested model core.
(Insert the architecture diagram and a runtime/cost table here.)

## Cost / runtime
| Stage | Hardware | Runtime | Approx cost |
|---|---|---|---|
| Batch job | CPU / L40S | TBD | TBD |
| Endpoint (demo) | CPU / L40S | TBD | TBD |

License: MIT.
