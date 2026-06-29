"""Contract tests for proposal 030 — semantic twin.

These tests assert STRUCTURAL invariants that must hold for ANY valid build.
They are the machine-readable analogue of the orphan-check crux described in
contracts/saref_mapping.md.

RED phase: all fail with ImportError until src/twin/ is implemented.
"""
from __future__ import annotations

import pytest
from contracts.schema import EnrichedPoint, EnrichedTrajectory, Prediction, TrajectoryPoint


# ---------------------------------------------------------------------------
# Helpers — build small, self-contained fixtures by hand.
# ---------------------------------------------------------------------------

def _make_enriched(
    agent_id: str,
    n_observed: int = 3,
    n_predicted: int = 2,
    anomaly_indices: tuple[int, ...] = (1,),
    lat_offset: float = 0.0,
) -> EnrichedTrajectory:
    """Minimal EnrichedTrajectory fixture independent of src/data or src/model."""
    observed = [
        EnrichedPoint(
            t=float(i),
            lat=52.0 + lat_offset + i * 0.01,
            lon=5.0 + i * 0.01,
            anomaly_score=0.9 if i in anomaly_indices else 0.1,
            is_anomaly=(i in anomaly_indices),
        )
        for i in range(n_observed)
    ]
    pred_points = [
        TrajectoryPoint(
            t=float(n_observed + j),
            lat=52.0 + lat_offset + (n_observed + j) * 0.01,
            lon=5.0 + (n_observed + j) * 0.01,
        )
        for j in range(n_predicted)
    ]
    return EnrichedTrajectory(
        agent_id=agent_id,
        observed=observed,
        prediction=Prediction(
            agent_id=agent_id,
            points=pred_points,
            anomaly_score=0.9 if anomaly_indices else 0.1,
            is_anomaly=bool(anomaly_indices),
        ),
    )


# ---------------------------------------------------------------------------
# Acceptance query strings — imported from src/twin/query.py (the canonical
# source that 050 and the Live DoD will also use). If these drift, this
# import will catch the divergence.
# ---------------------------------------------------------------------------
# Deferred import so the file can still be parsed when src/twin/ doesn't exist.
def _orphan_check() -> str:
    from src.twin.query import ORPHAN_CHECK_QUERY
    return ORPHAN_CHECK_QUERY


# ---------------------------------------------------------------------------
# Contract test 1: orphan-check SPARQL returns ZERO rows on any valid build.
# ---------------------------------------------------------------------------

class TestOrphanCheckEmpty:
    """The orphan-check invariant: every saref:Observation has an observedAgent.

    Each test asserts BOTH that Observation nodes exist (non-vacuous) AND that
    the orphan-check query returns zero rows.  A query typed on saref:Observation
    against a graph with zero Observation nodes would return empty vacuously and
    give a false green — the node-count assertion prevents that.
    """

    def test_single_agent_no_orphans(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = _make_enriched("agent_1", n_observed=3, n_predicted=2)
        g = build_graph([traj])

        # Non-vacuous: confirm saref:Observation nodes actually exist (3+2=5)
        q_count = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT (COUNT(?o) AS ?n) WHERE { ?o a saref:Observation }"
        )
        count_rows = sparql(g, q_count)
        assert int(count_rows[0]["n"]) == 5, "Expected 5 Observation nodes"

        # Then assert orphan check is zero (not vacuously empty)
        rows = sparql(g, _orphan_check())
        assert rows == [], f"Orphan check returned rows: {rows}"

    def test_multi_agent_no_orphans(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        # 5 agents, each with 3 observed + 2 predicted = 5 obs each → 25 total
        trajs = [_make_enriched(f"agent_{i}", lat_offset=i * 1.0) for i in range(5)]
        g = build_graph(trajs)

        # Non-vacuous: 5 agents × 5 nodes each = 25 saref:Observation nodes
        q_count = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT (COUNT(?o) AS ?n) WHERE { ?o a saref:Observation }"
        )
        count_rows = sparql(g, q_count)
        assert int(count_rows[0]["n"]) == 25, "Expected 25 Observation nodes across 5 agents"

        rows = sparql(g, _orphan_check())
        assert rows == [], f"Orphan check returned rows: {rows}"

    def test_zero_observations_empty_prediction_no_orphans(self):
        """Agent with no observations AND no predicted points → no orphans."""
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = EnrichedTrajectory(
            agent_id="empty_agent",
            observed=[],
            prediction=Prediction(
                agent_id="empty_agent",
                points=[],
                anomaly_score=0.0,
                is_anomaly=False,
            ),
        )
        g = build_graph([traj])
        rows = sparql(g, _orphan_check())
        assert rows == []


# ---------------------------------------------------------------------------
# Contract test 2: every agent with predicted points also has observed points.
# ---------------------------------------------------------------------------

class TestPredictedImpliesObserved:
    """Provenance invariant: no predicted measurement exists without observed ones."""

    def test_predicted_without_observed_raises(self):
        """build_graph MUST raise when prediction.points is non-empty but observed is empty."""
        from src.twin.graph import build_graph

        bad = EnrichedTrajectory(
            agent_id="ghost",
            observed=[],  # no history
            prediction=Prediction(
                agent_id="ghost",
                points=[TrajectoryPoint(t=1.0, lat=52.0, lon=5.0)],
                anomaly_score=0.5,
                is_anomaly=False,
            ),
        )
        with pytest.raises(ValueError, match="ghost"):
            build_graph([bad])

    def test_valid_agent_has_both(self):
        """A normal trajectory: observed non-empty, predicted non-empty → OK."""
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = _make_enriched("ok_agent")
        g = build_graph([traj])

        q_has_observed = (
            "PREFIX ex: <http://example.org/twin/> "
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?a WHERE { "
            "  ?obs a saref:Observation ; ex:observedAgent ?a ; ex:kind \"observed\" "
            "} GROUP BY ?a"
        )
        q_has_predicted = (
            "PREFIX ex: <http://example.org/twin/> "
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?a WHERE { "
            "  ?pred a saref:Observation ; ex:observedAgent ?a ; ex:kind \"predicted\" "
            "} GROUP BY ?a"
        )
        observed_agents = {r["a"] for r in sparql(g, q_has_observed)}
        predicted_agents = {r["a"] for r in sparql(g, q_has_predicted)}
        # Every agent with predicted points must also have observed points
        assert predicted_agents <= observed_agents, (
            f"Predicted-only agents: {predicted_agents - observed_agents}"
        )

    def test_multi_agent_predicted_implies_observed(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        trajs = [_make_enriched(f"ag_{i}", lat_offset=i * 0.5) for i in range(4)]
        g = build_graph(trajs)

        q_obs = (
            "PREFIX ex: <http://example.org/twin/> "
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?a WHERE { "
            "  ?o a saref:Observation ; ex:observedAgent ?a ; ex:kind \"observed\" "
            "} GROUP BY ?a"
        )
        q_pred = (
            "PREFIX ex: <http://example.org/twin/> "
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?a WHERE { "
            "  ?p a saref:Observation ; ex:observedAgent ?a ; ex:kind \"predicted\" "
            "} GROUP BY ?a"
        )
        observed_agents = {r["a"] for r in sparql(g, q_obs)}
        predicted_agents = {r["a"] for r in sparql(g, q_pred)}
        assert predicted_agents <= observed_agents


# ---------------------------------------------------------------------------
# Contract test 3: agent IRI and kind are correct.
# ---------------------------------------------------------------------------

class TestTripleShapes:
    """The emitted triples conform to the saref_mapping.md spec."""

    def test_agent_is_saref_device(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql
        import rdflib

        traj = _make_enriched("dev_1")
        g = build_graph([traj])
        q = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?a WHERE { ?a a saref:Device }"
        )
        rows = sparql(g, q)
        assert len(rows) == 1
        assert str(rows[0]["a"]) == "http://example.org/twin/Agent_dev_1"

    def test_observation_count(self):
        """Observed + predicted counts match fixture sizes (saref:Observation v2)."""
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = _make_enriched("cnt", n_observed=3, n_predicted=2)
        g = build_graph([traj])
        q = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT (COUNT(?m) AS ?n) WHERE { ?m a saref:Observation }"
        )
        rows = sparql(g, q)
        assert int(rows[0]["n"]) == 5  # 3 observed + 2 predicted
