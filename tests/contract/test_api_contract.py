"""Contract tests for proposal 050 — twin_api FastAPI endpoint.

Tests are written against contracts/openapi.yaml and run IN-PROCESS via
Starlette TestClient (no port binding — satisfies the parallel-agent constraint).
TestClient used as a context manager triggers lifespan so startup loads the
residual and twin graph correctly.

All four routes are tested:
    GET  /health       – no auth
    POST /predict      – bearer auth, inference
    GET  /twin/state   – bearer auth, twin query
    POST /twin/sparql  – bearer auth, read-only SPARQL
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import pytest
from starlette.testclient import TestClient

from contracts.schema import (
    HISTORY_LEN,
    HORIZON,
    EnrichedPoint,
    EnrichedTrajectory,
    Prediction,
    TrajectoryPoint,
)

# ---------------------------------------------------------------------------
# Fixture: build a tiny twin graph saved to a tmp Turtle file
# ---------------------------------------------------------------------------

_AGENT_ID = "test_agent_1"
_TEST_TOKEN = "test-secret-token-xyz"


def _make_enriched(agent_id: str) -> EnrichedTrajectory:
    """Minimal EnrichedTrajectory with 3 observed + 2 predicted points."""
    observed = [
        EnrichedPoint(
            t=float(i),
            lat=52.0 + i * 0.01,
            lon=5.0 + i * 0.01,
            anomaly_score=0.1,
            is_anomaly=False,
        )
        for i in range(3)
    ]
    pred_points = [
        TrajectoryPoint(
            t=float(3 + j),
            lat=52.03 + j * 0.01,
            lon=5.03 + j * 0.01,
        )
        for j in range(2)
    ]
    return EnrichedTrajectory(
        agent_id=agent_id,
        observed=observed,
        prediction=Prediction(
            agent_id=agent_id,
            points=pred_points,
            anomaly_score=0.1,
            is_anomaly=False,
        ),
    )


@pytest.fixture(scope="module")
def twin_ttl_path(tmp_path_factory) -> pathlib.Path:
    """Build a real twin graph and save to a tmp Turtle file."""
    from src.twin.graph import build_graph
    from src.twin.store import save_turtle

    traj = _make_enriched(_AGENT_ID)
    graph = build_graph([traj])

    tmp_dir = tmp_path_factory.mktemp("twin")
    path = tmp_dir / "twin.ttl"
    save_turtle(graph, path)
    return path


@pytest.fixture(scope="module")
def client(twin_ttl_path):
    """TestClient with lifespan, env vars set before context entry."""
    # Set env vars BEFORE the app loads so lifespan sees them
    os.environ["ENDPOINT_AUTH_TOKEN"] = _TEST_TOKEN
    os.environ["TWIN_TTL_PATH"] = str(twin_ttl_path)
    os.environ.pop("MODEL_PATH", None)  # no residual model in tests

    from services.twin_api.app import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def auth_headers(token: str = _TEST_TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


def valid_history(length: int = HISTORY_LEN) -> list[list[float]]:
    """Generate a valid history of given length."""
    return [
        [float(i), 52.0 + i * 0.001, 5.0 + i * 0.001, 0.001, 0.001]
        for i in range(length)
    ]


# ---------------------------------------------------------------------------
# 1. GET /health — no auth required
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "twin_loaded" in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_twin_loaded_true(self, client):
        """Twin was loaded from the fixture path, so twin_loaded must be True."""
        data = client.get("/health").json()
        assert data["twin_loaded"] is True

    def test_health_model_loaded_true(self, client):
        """load_residual never raises; returns _ZeroResidual, so model_loaded is True."""
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_no_auth_needed(self, client):
        """No Authorization header required for /health."""
        resp = client.get("/health")  # no headers
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_valid_returns_200(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": valid_history(HISTORY_LEN)},
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_predict_response_has_required_keys(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": valid_history(HISTORY_LEN)},
            headers=auth_headers(),
        )
        data = resp.json()
        assert "agent_id" in data
        assert "prediction" in data
        assert "anomaly_score" in data
        assert "is_anomaly" in data

    def test_predict_prediction_length_equals_horizon(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": valid_history(HISTORY_LEN)},
            headers=auth_headers(),
        )
        data = resp.json()
        assert len(data["prediction"]) == HORIZON

    def test_predict_each_prediction_row_has_5_elements(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": valid_history(HISTORY_LEN)},
            headers=auth_headers(),
        )
        for row in resp.json()["prediction"]:
            assert len(row) == 5

    def test_predict_anomaly_score_in_range(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": valid_history(HISTORY_LEN)},
            headers=auth_headers(),
        )
        score = resp.json()["anomaly_score"]
        assert 0.0 <= score <= 1.0

    def test_predict_is_anomaly_is_bool(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": valid_history(HISTORY_LEN)},
            headers=auth_headers(),
        )
        assert isinstance(resp.json()["is_anomaly"], bool)

    def test_predict_agent_id_echoed(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "my_vehicle", "history": valid_history(HISTORY_LEN)},
            headers=auth_headers(),
        )
        assert resp.json()["agent_id"] == "my_vehicle"

    def test_predict_more_than_history_len_ok(self, client):
        """Extra history rows are fine."""
        resp = client.post(
            "/predict",
            json={"agent_id": "a", "history": valid_history(HISTORY_LEN + 5)},
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_predict_history_too_short_returns_422(self, client):
        short = valid_history(HISTORY_LEN - 1)
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": short},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    def test_predict_empty_history_returns_422(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": []},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    def test_predict_no_token_returns_401(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": valid_history(HISTORY_LEN)},
            # no Authorization header
        )
        assert resp.status_code == 401

    def test_predict_wrong_token_returns_401(self, client):
        resp = client.post(
            "/predict",
            json={"agent_id": "agent_x", "history": valid_history(HISTORY_LEN)},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_predict_malformed_row_returns_422(self, client):
        """A row with 4 elements (not 5) must be rejected."""
        bad_history = [[1.0, 52.0, 5.0, 0.001]] * HISTORY_LEN  # only 4 elements
        resp = client.post(
            "/predict",
            json={"agent_id": "a", "history": bad_history},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    def test_predict_row_with_6_elements_returns_422(self, client):
        """A row with 6 elements (not exactly 5) must be rejected (maxItems:5 per openapi)."""
        bad_history = [[1.0, 52.0, 5.0, 0.001, 0.001, 99.9]] * HISTORY_LEN  # 6 elements
        resp = client.post(
            "/predict",
            json={"agent_id": "a", "history": bad_history},
            headers=auth_headers(),
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. GET /twin/state
# ---------------------------------------------------------------------------

class TestTwinState:
    def test_known_agent_returns_200(self, client):
        resp = client.get(
            f"/twin/state?agent_id={_AGENT_ID}",
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_known_agent_has_required_keys(self, client):
        resp = client.get(
            f"/twin/state?agent_id={_AGENT_ID}",
            headers=auth_headers(),
        )
        data = resp.json()
        assert "agent_id" in data
        assert "latest" in data
        assert "predicted" in data

    def test_known_agent_id_echoed(self, client):
        resp = client.get(
            f"/twin/state?agent_id={_AGENT_ID}",
            headers=auth_headers(),
        )
        assert resp.json()["agent_id"] == _AGENT_ID

    def test_known_agent_latest_not_none(self, client):
        resp = client.get(
            f"/twin/state?agent_id={_AGENT_ID}",
            headers=auth_headers(),
        )
        assert resp.json()["latest"] is not None

    def test_known_agent_predicted_not_none(self, client):
        resp = client.get(
            f"/twin/state?agent_id={_AGENT_ID}",
            headers=auth_headers(),
        )
        assert resp.json()["predicted"] is not None

    def test_unknown_agent_returns_404(self, client):
        resp = client.get(
            "/twin/state?agent_id=no_such_agent_xyz",
            headers=auth_headers(),
        )
        assert resp.status_code == 404

    def test_no_token_returns_401(self, client):
        resp = client.get(f"/twin/state?agent_id={_AGENT_ID}")
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, client):
        resp = client.get(
            f"/twin/state?agent_id={_AGENT_ID}",
            headers={"Authorization": "Bearer bad-token"},
        )
        assert resp.status_code == 401

    # --- SSRF / injection regression tests ----------------------------------

    def test_injection_payload_returns_404_not_500(self, client):
        """Confirmed injection payload: agent_id containing '>' + SERVICE clause.

        Live payload that triggered outbound connection (URLError Errno 111):
            victim> ?p ?o . SERVICE <http://127.0.0.1:9/sparql> { ?s ?x ?y } } #

        With the initBindings fix the value is a URIRef VALUE (not query text)
        so it matches nothing in the graph → 404; no outbound attempt is made.
        """
        payload = "victim> ?p ?o . SERVICE <http://127.0.0.1:9/sparql> { ?s ?x ?y } } #"
        resp = client.get(
            f"/twin/state?agent_id={payload}",
            headers=auth_headers(),
        )
        # Must return 404 (unknown agent), never 500 (injection / outbound)
        assert resp.status_code == 404

    def test_injection_payload_no_outbound_call(self, client, monkeypatch):
        """Injection payload must NOT trigger outbound network access.

        Spies on rdflib.Graph.query to assert it receives PreparedQuery objects
        (not raw injected strings), proving no SERVICE text entered the engine.
        """
        import rdflib

        original_query = rdflib.Graph.query
        received: list = []

        def spy_query(self, query_object, *args, **kwargs):
            received.append(query_object)
            return original_query(self, query_object, *args, **kwargs)

        monkeypatch.setattr(rdflib.Graph, "query", spy_query)

        payload = "victim> ?p ?o . SERVICE <http://127.0.0.1:9/sparql> { ?s ?x ?y } } #"
        resp = client.get(
            f"/twin/state?agent_id={payload}",
            headers=auth_headers(),
        )
        assert resp.status_code == 404

        # Every graph.query() call must have been passed a PreparedQuery
        # (has .algebra), not a raw string.
        for q in received:
            assert hasattr(q, "algebra"), (
                f"graph.query received a raw string instead of PreparedQuery: {q!r}"
            )

    def test_angle_bracket_agent_id_returns_404(self, client):
        """agent_id containing '>' alone → 404 (treated as unusual IRI, not injection)."""
        resp = client.get(
            "/twin/state?agent_id=foo%3Ebar",  # URL-encoded '>'
            headers=auth_headers(),
        )
        assert resp.status_code == 404

    def test_normal_agent_still_200_after_injection_fix(self, client):
        """Regression: legitimate agent still returns 200 after the parameterized fix."""
        resp = client.get(
            f"/twin/state?agent_id={_AGENT_ID}",
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_unknown_agent_still_404_after_injection_fix(self, client):
        """Regression: genuinely unknown agent still returns 404."""
        resp = client.get(
            "/twin/state?agent_id=completely_unknown_xyz",
            headers=auth_headers(),
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. POST /twin/sparql
# ---------------------------------------------------------------------------

class TestTwinSparql:
    def test_valid_select_returns_200(self, client):
        resp = client.post(
            "/twin/sparql",
            json={"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_valid_select_has_head_and_results(self, client):
        resp = client.post(
            "/twin/sparql",
            json={"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"},
            headers=auth_headers(),
        )
        data = resp.json()
        assert "head" in data
        assert "results" in data

    def test_select_against_real_graph(self, client):
        """Query that should find our test agent's Device node."""
        q = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?a WHERE { ?a a saref:Device }"
        )
        resp = client.post(
            "/twin/sparql",
            json={"query": q},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        bindings = data["results"]["bindings"]
        assert len(bindings) >= 1

    def test_insert_rejected_with_400(self, client):
        """SPARQL INSERT (mutation) must return 400."""
        q = (
            "PREFIX ex: <http://example.org/twin/> "
            "INSERT DATA { ex:foo ex:bar ex:baz }"
        )
        resp = client.post(
            "/twin/sparql",
            json={"query": q},
            headers=auth_headers(),
        )
        assert resp.status_code == 400

    def test_delete_rejected_with_400(self, client):
        """SPARQL DELETE must return 400."""
        q = (
            "DELETE WHERE { ?s ?p ?o }"
        )
        resp = client.post(
            "/twin/sparql",
            json={"query": q},
            headers=auth_headers(),
        )
        assert resp.status_code == 400

    def test_service_clause_rejected_with_400(self, client):
        """SERVICE clause (SSRF vector) must be rejected via algebra walk (not string match).

        The algebra is parsed with prepareQuery and walked for ServiceGraphPattern.
        No outbound connection is made — a fast 400 is the evidence.
        """
        q = "SELECT * WHERE { SERVICE <http://evil.example.com/sparql> { ?s ?p ?o } }"
        resp = client.post(
            "/twin/sparql",
            json={"query": q},
            headers=auth_headers(),
        )
        assert resp.status_code == 400

    def test_service_rejected_before_execution(self, client, monkeypatch):
        """SERVICE must be rejected BEFORE any graph execution (no outbound attempt).

        Monkeypatching twin_sparql in the app module proves graph.query() is
        never called, so no outbound federation can occur.
        """
        import services.twin_api.app as appmod

        def boom(*a, **k):
            raise AssertionError("twin_sparql was called — SERVICE clause not rejected pre-execution")

        monkeypatch.setattr(appmod, "twin_sparql", boom)
        q = "SELECT * WHERE { SERVICE <http://evil.example.com/sparql> { ?s ?p ?o } }"
        resp = client.post(
            "/twin/sparql",
            json={"query": q},
            headers=auth_headers(),
        )
        assert resp.status_code == 400

    def test_service_in_union_rejected_with_400(self, client):
        """SERVICE nested inside UNION must also be caught by the recursive walk."""
        q = (
            "SELECT * WHERE { "
            "  { ?a ?b ?c } "
            "  UNION { SERVICE <http://evil.example.com/> { ?s ?p ?o } } "
            "}"
        )
        resp = client.post(
            "/twin/sparql",
            json={"query": q},
            headers=auth_headers(),
        )
        assert resp.status_code == 400

    def test_ask_query_returns_200(self, client):
        resp = client.post(
            "/twin/sparql",
            json={"query": "ASK { ?s ?p ?o }"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_ask_response_has_head_and_results(self, client):
        resp = client.post(
            "/twin/sparql",
            json={"query": "ASK { ?s ?p ?o }"},
            headers=auth_headers(),
        )
        data = resp.json()
        assert "head" in data
        assert "results" in data

    def test_no_token_returns_401(self, client):
        resp = client.post(
            "/twin/sparql",
            json={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
        )
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, client):
        resp = client.post(
            "/twin/sparql",
            json={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_parse_error_returns_400(self, client):
        resp = client.post(
            "/twin/sparql",
            json={"query": "THIS IS NOT VALID SPARQL !!!"},
            headers=auth_headers(),
        )
        assert resp.status_code == 400

    def test_head_vars_present_for_select(self, client):
        q = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?a WHERE { ?a a saref:Device }"
        )
        resp = client.post(
            "/twin/sparql",
            json={"query": q},
            headers=auth_headers(),
        )
        data = resp.json()
        assert "vars" in data["head"]
        assert "a" in data["head"]["vars"]

    def test_bindings_present_for_select(self, client):
        q = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?a WHERE { ?a a saref:Device }"
        )
        resp = client.post(
            "/twin/sparql",
            json={"query": q},
            headers=auth_headers(),
        )
        data = resp.json()
        assert "bindings" in data["results"]
