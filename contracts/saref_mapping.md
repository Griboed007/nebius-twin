# SAREF / RDF mapping contract (twin layer) — v2

SOURCE OF TRUTH for how `EnrichedTrajectory` becomes an RDF graph and what
SPARQL must return. Corrected per independent verification against SAREF core
v3.2.1 / v4.1.1: `saref:Measurement` is deprecated in favour of
`saref:Observation` (SOSA/SSN convergence). Use `saref:Observation`.

## Namespaces

| prefix | IRI |
|---|---|
| `saref` | `https://saref.etsi.org/core/` |
| `geo`   | `http://www.w3.org/2003/01/geo/wgs84_pos#` |
| `xsd`   | `http://www.w3.org/2001/XMLSchema#` |
| `ex`    | `http://example.org/twin/` |

(Optional, if adopting SAREF4AUTO — see proposal 031 Design:
`s4auto` = `https://saref.etsi.org/saref4auto/`.)

## Entities

**Agent** — one tracked vehicle.
- `ex:Agent_{agent_id}` `a` `saref:Device` .
- `ex:Agent_{agent_id}` `ex:agentId` `"{agent_id}"^^xsd:string` .

**Observation** — one observed or predicted state. (Was `saref:Measurement`.)
- `ex:Obs_{agent_id}_{t}` `a` `saref:Observation` .
- `saref:hasTimestamp` `"{iso8601}"^^xsd:dateTime` .
  (SOSA-convergent alternatives if desired: `saref:hasResultTime` /
  `saref:hasPhenomenonTime`. `hasTimestamp` retained here for minimal change.)
- `geo:lat` `"{lat}"^^xsd:double` ; `geo:long` `"{lon}"^^xsd:double` .
- `ex:observedAgent` `ex:Agent_{agent_id}` .            # provenance link
- `ex:kind` `"observed"` or `"predicted"` .
- (observed only) `ex:anomalyScore` `"{score}"^^xsd:double` ; `ex:isAnomaly` `"{bool}"^^xsd:boolean` .

## Provenance invariant (the verification crux) — UNCHANGED

Every `saref:Observation` node MUST carry exactly one `ex:observedAgent` whose
object is a declared `saref:Device`. Zero orphan observations; zero predicted
observations whose `agent_id` has no observed observations in the same graph.
Enforced at build time so the orphan-check query is empty by construction.

## Required SPARQL behaviours (acceptance queries)

1. **Anomaly rollup** — count anomalous observations per agent (unchanged):
   `SELECT ?a (COUNT(?o) AS ?n) WHERE { ?o ex:observedAgent ?a ; ex:isAnomaly true } GROUP BY ?a`
2. **Geofence-style filter** — predicted points outside a lat/lon box (unchanged):
   a SELECT over `ex:kind "predicted"` filtered by `geo:lat`/`geo:long` bounds.
3. **Orphan check (MUST return empty)** — now types on `saref:Observation`:
   `SELECT ?o WHERE { ?o a saref:Observation . FILTER NOT EXISTS { ?o ex:observedAgent ?a } }`

The endpoint's `/twin/sparql` MUST accept arbitrary read-only SELECT/ASK and
reject UPDATE/INSERT/DELETE **and any SERVICE (federation) clause** — see
proposal 050's SSRF requirement.