# Semantic Digital Twin for Vehicle Telematics on Nebius Serverless

A reproducible Physical-AI pipeline for the Nebius Serverless AI Builders
Challenge. A Nebius Serverless **Job** enriches GPS trajectories with hybrid
trajectory prediction + anomaly scoring and writes a **SAREF/RDF** twin; a Nebius
Serverless **Endpoint** serves live predictions and **SPARQL** queries over it.

The differentiator: the twin is *queryable and explainable*. "Which agents are
predicted to breach this geofence?" is a SPARQL query over a SAREF graph
populated by serverless ML — not a black-box score.

## Reproduce offline (no downloads)

```
pip install -r requirements.txt
make synth && make test && make job
```

Produces `enriched.parquet`, `twin.ttl`, and `metrics.json` in `artifacts/out/`.
The synthetic generator is deterministic and ships ground-truth anomaly labels,
so the pipeline is fully reproducible with no external data.

## Run on Nebius

Set `.env` from `.env.example`, then:

```
make preflight        # create-if-missing bucket, seed input
make push             # build + push job and endpoint images
DEPLOY_CONFIRM=1 bash deploy/job.sh <job_image>
DEPLOY_CONFIRM=1 bash deploy/endpoint.sh <endpoint_image>
make teardown         # delete endpoint + job when done
```

Live proof of execution is in `docs/proof/`.

## Architecture

`Sensor data → Batch Job (enrich) → object storage (SAREF/RDF twin) → Inference
Endpoint`, with both services loading one shared, tested model core. See
`docs/architecture.svg`.

- **Model** — constant-velocity baseline + optional learned residual (CPU-safe);
  anomaly score = normalised prediction error.
- **Twin** — `saref:Device` agents, `saref:Observation` states (current SAREF
  core term), provenance invariant enforced at build time (zero orphans by
  construction), read-only SPARQL with federation (`SERVICE`) rejected.

## Results (live run on Nebius)

| Stage | Hardware | Runtime | Output / cost |
|---|---|---|---|
| Batch enrichment Job | cpu-e2 (2 vCPU, 8 GB) | 1.9 s compute | 3,000 observations → `enriched.parquet` + `twin.ttl` (24,380 triples) + `metrics.json`; a few cents (per-second billing) |
| Inference Endpoint | cpu-e2 (2 vCPU, 8 GB) | cold-start + seconds/req | `/predict` (6-step) + `/twin/sparql` over HTTPS; stopped after demo to pause billing |

- **Anomaly detection:** ROC-AUC **0.8247** on the labelled synthetic set (3,000
  observations, 218 flagged across 10 agents).
- **Twin integrity:** 3,060 `saref:Observation` nodes, 0 deprecated
  `saref:Measurement`, **0 orphans** on the cloud-produced `twin.ttl`.
- **Endpoint:** token auth enforced (401 without / 200 with); live SPARQL anomaly
  rollup matches batch results; geofence query returns 24 out-of-box predictions.

## License

MIT — see `LICENSE`.