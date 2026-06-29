# SAREF / RDF mapping contract (twin layer)

SOURCE OF TRUTH for how `EnrichedTrajectory` becomes an RDF graph and what
SPARQL must return. The implementer of `src/twin/` satisfies this; contract
tests assert the triples below exist. Verify exact SAREF term IRIs against the
live ontology via Context7 before finalising — flagged in the proposal's
"honest hard parts".

## Namespaces

| prefix | IRI |
|---|---|
| `saref` | `https://saref.etsi.org/core/` |
| `geo`   | `http://www.w3.org/2003/01/geo/wgs84_pos#` |
| `xsd`   | `http://www.w3.org/2001/XMLSchema#` |
| `ex`    | `http://example.org/twin/` |

## Entities

**Agent** — one tracked vehicle.
- `ex:Agent_{agent_id}` `a` `saref:Device` .
- `ex:Agent_{agent_id}` `ex:agentId` `"{agent_id}"^^xsd:string` .

**Observation** — one observed or predicted state.
- `ex:Obs_{agent_id}_{t}` `a` `saref:Measurement` .
- `saref:hasTimestamp` `"{iso8601}"^^xsd:dateTime` .
- `geo:lat` `"{lat}"^^xsd:double` ; `geo:long` `"{lon}"^^xsd:double` .
- `ex:observedAgent` `ex:Agent_{agent_id}` .            # provenance link
- `ex:kind` `"observed"` or `"predicted"` .
- (observed only) `ex:anomalyScore` `"{score}"^^xsd:double` ; `ex:isAnomaly` `"{bool}"^^xsd:boolean` .

## Provenance invariant (the verification crux)

Every `saref:Measurement` node MUST carry exactly one `ex:observedAgent` whose
object is a declared `saref:Device`. There SHALL be zero orphan measurements
(a node with no parent agent) and zero predicted measurements whose `agent_id`
has no observed measurements in the same graph. This is the structural analogue
of deckgen's "every figure resolves to a ledger row, zero dangling," and it is
what the Strategist checks read-only with SPARQL after the batch job runs.

## Required SPARQL behaviours (acceptance queries)

1. **Anomaly rollup** — count anomalous observations per agent:
   `SELECT ?a (COUNT(?o) AS ?n) WHERE { ?o ex:observedAgent ?a ; ex:isAnomaly true } GROUP BY ?a`
2. **Geofence-style filter** — predicted points outside a lat/lon box:
   a SELECT over `ex:kind "predicted"` filtered by `geo:lat`/`geo:long` bounds.
3. **Orphan check (MUST return empty)**:
   `SELECT ?o WHERE { ?o a saref:Measurement . FILTER NOT EXISTS { ?o ex:observedAgent ?a } }`

The endpoint's `/twin/sparql` MUST accept arbitrary read-only SELECT/ASK and
reject UPDATE/INSERT/DELETE with HTTP 400.
