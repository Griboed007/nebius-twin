"""FastAPI application for proposal 050 — semantic twin inference endpoint.

Implements exactly the four routes from contracts/openapi.yaml (READ-ONLY source
of truth):

    GET  /health       – liveness probe, no auth
    POST /predict      – bearer auth, trajectory → prediction + anomaly score
    GET  /twin/state   – bearer auth, agent state from twin graph
    POST /twin/sparql  – bearer auth, read-only SELECT/ASK (SERVICE blocked)

Auth: all routes except /health require Authorization: Bearer <token> where the
expected token is read from env ENDPOINT_AUTH_TOKEN. Missing or wrong token → 401.
Uses HTTPBearer(auto_error=False) to avoid FastAPI's default 403 for missing headers.

Startup (lifespan):
    - load_residual(MODEL_PATH)   → residual model (always succeeds, falls back to
                                    _ZeroResidual if MODEL_PATH unset or bad)
    - load_turtle(TWIN_TTL_PATH)  → rdflib.Graph (falls back to empty Graph if
                                    file absent, serving twin routes gracefully)

SERVICE detection: /twin/sparql parses the query algebra via rdflib's
prepareQuery and recursively walks CompValue nodes, rejecting any query that
contains a ServiceGraphPattern node — detected by algebra name, not string match.
"""
from __future__ import annotations

import os
import pathlib
from contextlib import asynccontextmanager
from typing import Any

import rdflib
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue

from contracts.schema import HISTORY_LEN, Trajectory, TrajectoryPoint
from src.model import core as model_core
from src.model.residual import load_residual
from src.twin.query import QueryRejectedError
from src.twin.query import sparql as twin_sparql
from src.twin.store import load_turtle

# ---------------------------------------------------------------------------
# Application state — populated by lifespan, read per-request
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "residual": None,
    "twin_graph": None,
    "twin_loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the residual model and twin graph once at startup."""
    # Residual: load_residual never raises (falls back to _ZeroResidual)
    _state["residual"] = load_residual(os.environ.get("MODEL_PATH"))

    # Twin graph: gracefully degrade if file is absent
    twin_path_str = os.environ.get("TWIN_TTL_PATH", "./artifacts/twin.ttl")
    twin_path = pathlib.Path(twin_path_str)
    try:
        _state["twin_graph"] = load_turtle(twin_path)
        _state["twin_loaded"] = True
    except FileNotFoundError:
        _state["twin_graph"] = rdflib.Graph()  # empty graph — twin routes return 404/empty
        _state["twin_loaded"] = False

    yield
    # No teardown needed — graphs are in-memory


app = FastAPI(
    title="Semantic Twin Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

# auto_error=False: missing header returns None instead of raising 403
_bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Raise 401 if Authorization: Bearer <token> is missing or invalid."""
    expected = os.environ.get("ENDPOINT_AUTH_TOKEN", "")
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# SERVICE detection — algebra walk, NOT string matching (SSRF prevention)
# ---------------------------------------------------------------------------

def _has_service_clause(node: Any) -> bool:
    """Return True if the algebra tree contains any ServiceGraphPattern node.

    Walks CompValue nodes recursively (handles nesting in UNION, OPTIONAL,
    sub-SELECT, etc.) without relying on string search of the raw query text.
    """
    if isinstance(node, CompValue):
        if node.name == "ServiceGraphPattern":
            return True
        for child in node.values():
            if _has_service_clause(child):
                return True
    elif isinstance(node, (list, tuple)):
        for item in node:
            if _has_service_clause(item):
                return True
    return False


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    agent_id: str
    history: list[list[float]]


class SparqlRequest(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# GET /health — no auth
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe. Returns model_loaded and twin_loaded flags."""
    return {
        "status": "ok",
        "model_loaded": _state["residual"] is not None,
        "twin_loaded": _state["twin_loaded"],
    }


# ---------------------------------------------------------------------------
# POST /predict — bearer auth required
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict(
    _auth: None = Depends(require_auth),
    body: PredictRequest = ...,
):
    """Predict HORIZON future states and compute anomaly score.

    Auth is checked first (Depends runs before body handling). 401 fires
    before any inference or 422 business logic.
    """
    # Validate history length (business rule, not Pydantic shape)
    if len(body.history) < HISTORY_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"history must have at least {HISTORY_LEN} rows; got {len(body.history)}",
        )

    # Validate each row is exactly 5 numbers (per openapi: minItems:5, maxItems:5)
    if any(len(row) != 5 for row in body.history):
        raise HTTPException(
            status_code=422,
            detail="each history row must contain exactly 5 numbers [t, lat, lon, vlat, vlon]",
        )

    # Build Trajectory (Pydantic validates lat/lon bounds)
    try:
        points = [
            TrajectoryPoint(
                t=float(row[0]),
                lat=float(row[1]),
                lon=float(row[2]),
                vlat=float(row[3]),
                vlon=float(row[4]),
            )
            for row in body.history
        ]
    except (IndexError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Malformed history row: {exc}")

    traj = Trajectory(agent_id=body.agent_id, points=points)
    pred = model_core.predict(traj, residual_model=_state["residual"])

    return {
        "agent_id": pred.agent_id,
        "prediction": [
            [p.t, p.lat, p.lon, p.vlat, p.vlon] for p in pred.points
        ],
        "anomaly_score": pred.anomaly_score,
        "is_anomaly": pred.is_anomaly,
    }


# ---------------------------------------------------------------------------
# GET /twin/state — bearer auth required
# ---------------------------------------------------------------------------
# Fixed SPARQL query strings for /twin/state — NO user input ever interpolated
# into query text. The ?agent variable is bound via initBindings at execution
# time, so any agent_id value (including injection payloads) becomes a harmless
# URIRef value that simply matches nothing → 404.
#
# These are prepared once at module load (fast) and reused per request.

_STATE_ASK_Q = prepareQuery(
    "ASK { ?agent ?p ?o }"
)

_STATE_OBS_Q = prepareQuery(
    "PREFIX ex: <http://example.org/twin/> "
    "PREFIX saref: <https://saref.etsi.org/core/> "
    "PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> "
    "SELECT ?t ?lat ?lon WHERE { "
    "  ?obs a saref:Observation ; ex:observedAgent ?agent ; "
    '       ex:kind "observed" ; saref:hasTimestamp ?t ; '
    "       geo:lat ?lat ; geo:long ?lon . "
    "} ORDER BY DESC(?t) LIMIT 1"
)

_STATE_PRED_Q = prepareQuery(
    "PREFIX ex: <http://example.org/twin/> "
    "PREFIX saref: <https://saref.etsi.org/core/> "
    "PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> "
    "SELECT ?t ?lat ?lon WHERE { "
    "  ?pred a saref:Observation ; ex:observedAgent ?agent ; "
    '         ex:kind "predicted" ; saref:hasTimestamp ?t ; '
    "         geo:lat ?lat ; geo:long ?lon . "
    "} ORDER BY DESC(?t) LIMIT 1"
)


def _exec_bound(
    graph: rdflib.Graph,
    prepared,
    bindings: dict,
) -> list[dict[str, Any]]:
    """Execute a prepareQuery result with initBindings; return list[dict] rows.

    Mirrors src.twin.query.sparql's row-normalisation but accepts a pre-parsed
    query object and explicit bindings — no user text ever touches the query.
    """
    result = graph.query(prepared, initBindings=bindings)
    if result.type == "ASK":
        return [{"_ask": bool(result.askAnswer)}]
    rows: list[dict[str, Any]] = []
    for row in result:
        d: dict[str, Any] = {}
        for var, val in row.asdict().items():
            if val is not None and hasattr(val, "toPython"):
                d[var] = val.toPython()
            else:
                d[var] = val
        rows.append(d)
    return rows


@app.get("/twin/state")
def twin_state(
    agent_id: str = Query(..., description="Agent identifier"),
    _auth: None = Depends(require_auth),
):
    """Return the most-recent observed and predicted state for an agent.

    SSRF-safe: the agent IRI is passed as an initBinding value, never
    concatenated into query text. A hostile agent_id that contains `>`,
    SERVICE clauses, or any SPARQL syntax produces a URIRef that simply
    matches nothing in the graph → 404. No outbound connection is possible.
    Returns 404 if the agent_id is not present in the twin graph.
    """
    graph: rdflib.Graph = _state["twin_graph"]

    # Build agent IRI using the same pattern as src/twin/graph.py::_agent_iri.
    # Any content in agent_id is treated as a URIRef VALUE, not query text.
    agent_iri = rdflib.URIRef(f"http://example.org/twin/Agent_{agent_id}")
    bindings: dict[str, rdflib.URIRef] = {"agent": agent_iri}

    try:
        # ASK: does the agent node exist at all?
        ask_rows = _exec_bound(graph, _STATE_ASK_Q, bindings)
        if not ask_rows or not ask_rows[0].get("_ask"):
            raise HTTPException(status_code=404, detail=f"Unknown agent_id: {agent_id}")

        # Latest observed point
        obs_rows = _exec_bound(graph, _STATE_OBS_Q, bindings)
        latest = obs_rows[0] if obs_rows else None

        # Latest predicted point
        pred_rows = _exec_bound(graph, _STATE_PRED_Q, bindings)
        predicted = pred_rows[0] if pred_rows else None

    except HTTPException:
        raise
    except Exception as exc:
        # Guard against any unexpected rdflib error for edge-case agent_ids
        raise HTTPException(status_code=500, detail=f"Twin query error: {exc}")

    return {
        "agent_id": agent_id,
        "latest": latest,
        "predicted": predicted,
    }


# ---------------------------------------------------------------------------
# POST /twin/sparql — bearer auth required
# ---------------------------------------------------------------------------

def _sparql_rows_to_json(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Map sparql()'s list[dict] to SPARQL 1.1 JSON format {head, results}.

    For SELECT: head.vars = variable names; results.bindings = list of row dicts
    where each value is {"type": "literal"|"uri", "value": str(v)}.
    For ASK: head = {}, results = {"boolean": bool}.

    Note: sparql() returns .toPython() natives so URI-vs-literal type fidelity
    is best-effort (URIRef becomes str; datetime/int/float are Python types).
    """
    if rows and "_ask" in rows[0]:
        # ASK result
        return {
            "head": {},
            "results": {"boolean": rows[0]["_ask"]},
        }

    # SELECT result
    vars_list: list[str] = list(rows[0].keys()) if rows else []
    bindings = [
        {
            k: {
                "type": "uri" if isinstance(v, rdflib.URIRef) else "literal",
                "value": str(v),
            }
            for k, v in row.items()
        }
        for row in rows
    ]
    return {
        "head": {"vars": vars_list},
        "results": {"bindings": bindings},
    }


@app.post("/twin/sparql")
def sparql_query(
    _auth: None = Depends(require_auth),
    body: SparqlRequest = ...,
):
    """Run a read-only SPARQL SELECT or ASK over the twin graph.

    Rejects:
    - Non-SELECT/ASK (INSERT/DELETE/UPDATE/CONSTRUCT/DESCRIBE) → 400
    - Parse errors → 400
    - SERVICE clauses (SSRF) → 400 (detected via algebra walk)
    No outbound connection is ever made.
    """
    graph: rdflib.Graph = _state["twin_graph"]

    # Parse and walk algebra for SERVICE clauses BEFORE executing
    try:
        parsed = prepareQuery(body.query)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"SPARQL parse error: {exc}",
        )

    if _has_service_clause(parsed.algebra):
        raise HTTPException(
            status_code=400,
            detail="SERVICE clauses are not permitted (SSRF prevention)",
        )

    # Execute via twin_sparql (also enforces SELECT/ASK read-only)
    try:
        rows = twin_sparql(graph, body.query)
    except QueryRejectedError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _sparql_rows_to_json(rows)
