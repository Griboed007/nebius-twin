"""Build an RDF graph from EnrichedTrajectory instances.

SOURCE OF TRUTH for the triple structure: contracts/saref_mapping.md (v2).

Contract v2 note: ``saref:Observation`` (not ``saref:Measurement``, which was
deprecated in SAREF core v3.2.1 and deleted from the live ontology) is the
type used for all observation nodes.  ``saref:Device`` and ``saref:hasTimestamp``
are unchanged.

Provenance invariant (enforced at BUILD TIME, not post-hoc):
- Every ``saref:Observation`` node is emitted from the same code path that
  already wrote its parent ``saref:Device``, so orphan observations cannot
  arise by construction.
- If ``prediction.points`` is non-empty but ``observed`` is empty, the input
  is malformed and a ``ValueError`` is raised immediately — no partial graph
  is emitted, so no orphaned predicted observation can ever enter the store.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import rdflib
from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD

from contracts.schema import EnrichedTrajectory

# ---------------------------------------------------------------------------
# Namespaces (per contracts/saref_mapping.md)
# ---------------------------------------------------------------------------
SAREF = Namespace("https://saref.etsi.org/core/")
GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
XSD_NS = XSD  # rdflib's built-in XSD namespace
EX = Namespace("http://example.org/twin/")


def _unix_to_iso8601(t: float) -> str:
    """Convert UNIX seconds (float) to UTC ISO8601 string for xsd:dateTime."""
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _iri_safe_t(t: float) -> str:
    """Produce an IRI-safe token from a UNIX timestamp float."""
    # Replace '.' with '_' so floats like 0.5 become '0_5' in the IRI.
    return str(t).replace(".", "_")


def _agent_iri(agent_id: str) -> URIRef:
    return EX[f"Agent_{agent_id}"]


def _obs_iri(agent_id: str, t: float) -> URIRef:
    return EX[f"Obs_{agent_id}_{_iri_safe_t(t)}"]


def build_graph(trajectories: Sequence[EnrichedTrajectory]) -> Graph:
    """Build an rdflib.Graph from a sequence of EnrichedTrajectory records.

    Provenance invariant guarantee:
    - ``saref:Device`` for the agent is written FIRST.
    - Every ``saref:Observation`` references that device via ``ex:observedAgent``.
    - An agent whose prediction has points but has no observed points raises
      ``ValueError`` so no orphaned predicted observation is ever emitted.

    Parameters
    ----------
    trajectories:
        One entry per tracked vehicle.

    Returns
    -------
    rdflib.Graph
        An in-memory graph populated with the triple structure from the mapping.

    Raises
    ------
    ValueError
        If an EnrichedTrajectory has non-empty prediction.points but empty
        observed list (violates the provenance invariant).
    """
    # --- Pre-validate all inputs before touching the graph -------------------
    for traj in trajectories:
        if traj.prediction.points and not traj.observed:
            raise ValueError(
                f"Provenance violation: agent '{traj.agent_id}' has "
                f"{len(traj.prediction.points)} predicted point(s) but no "
                f"observed points. Refusing to emit orphaned measurements."
            )

    g = Graph()
    g.bind("saref", SAREF)
    g.bind("geo", GEO)
    g.bind("ex", EX)
    g.bind("xsd", XSD_NS)

    for traj in trajectories:
        agent_iri = _agent_iri(traj.agent_id)

        # --- Agent (saref:Device) ---
        g.add((agent_iri, RDF.type, SAREF.Device))
        g.add((agent_iri, EX.agentId, Literal(traj.agent_id, datatype=XSD_NS.string)))

        # --- Observed measurements ---
        for pt in traj.observed:
            obs_iri = _obs_iri(traj.agent_id, pt.t)
            g.add((obs_iri, RDF.type, SAREF.Observation))
            g.add((obs_iri, SAREF.hasTimestamp,
                   Literal(_unix_to_iso8601(pt.t), datatype=XSD_NS.dateTime)))
            g.add((obs_iri, GEO.lat,  Literal(pt.lat, datatype=XSD_NS.double)))
            g.add((obs_iri, GEO.long, Literal(pt.lon, datatype=XSD_NS.double)))
            g.add((obs_iri, EX.observedAgent, agent_iri))       # provenance link
            g.add((obs_iri, EX.kind, Literal("observed")))
            g.add((obs_iri, EX.anomalyScore,
                   Literal(pt.anomaly_score, datatype=XSD_NS.double)))
            g.add((obs_iri, EX.isAnomaly,
                   Literal(pt.is_anomaly, datatype=XSD_NS.boolean)))

        # --- Predicted measurements ---
        for pt in traj.prediction.points:
            obs_iri = _obs_iri(traj.agent_id, pt.t)
            g.add((obs_iri, RDF.type, SAREF.Observation))
            g.add((obs_iri, SAREF.hasTimestamp,
                   Literal(_unix_to_iso8601(pt.t), datatype=XSD_NS.dateTime)))
            g.add((obs_iri, GEO.lat,  Literal(pt.lat, datatype=XSD_NS.double)))
            g.add((obs_iri, GEO.long, Literal(pt.lon, datatype=XSD_NS.double)))
            g.add((obs_iri, EX.observedAgent, agent_iri))       # provenance link
            g.add((obs_iri, EX.kind, Literal("predicted")))
            # Note: no anomalyScore / isAnomaly on predicted points per the mapping

    return g
