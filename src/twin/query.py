"""Read-only SPARQL query interface for the semantic twin.

Only SELECT and ASK queries are permitted.  Any other form — UPDATE, INSERT,
DELETE, CONSTRUCT, DESCRIBE — is rejected BEFORE the graph is touched.

Two-layer read-only guarantee:
1. ``prepareQuery`` raises ``ParseException`` on all SPARQL Update forms, so
   the bad query never reaches the graph.
2. For syntactically-valid non-mutating forms (CONSTRUCT, DESCRIBE), the
   algebra name is checked; only ``SelectQuery`` and ``AskQuery`` are allowed.
3. ``sparql()`` calls only ``graph.query()``, never ``graph.update()``, so
   even if a query somehow passed the checks it could not mutate the graph.

ASK return shape:  ``[{"_ask": True}]`` or ``[{"_ask": False}]``.
SELECT return shape: ``list[dict]`` where values are ``.toPython()`` natives.

Acceptance query constants (reused by tests and the endpoint in 050):
    ANOMALY_ROLLUP_QUERY
    geofence_query(lat_min, lat_max, lon_min, lon_max) -> str
    ORPHAN_CHECK_QUERY

Warning for 050 (the endpoint layer):
    A read-only SELECT can contain a SERVICE clause that federates to an
    external SPARQL endpoint.  This is an SSRF vector.  The HTTP layer (050)
    MUST reject or strip SERVICE clauses before calling sparql().
"""
from __future__ import annotations

from typing import Any

import rdflib
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class QueryRejectedError(ValueError):
    """Raised when a non-SELECT/ASK query (or invalid SPARQL) is submitted."""


# ---------------------------------------------------------------------------
# Acceptance query constants
# ---------------------------------------------------------------------------

ANOMALY_ROLLUP_QUERY: str = (
    "PREFIX ex: <http://example.org/twin/> "
    "SELECT ?a (COUNT(?o) AS ?n) WHERE { "
    "  ?o ex:observedAgent ?a ; ex:isAnomaly true "
    "} GROUP BY ?a"
)

ORPHAN_CHECK_QUERY: str = (
    "PREFIX saref: <https://saref.etsi.org/core/> "
    "PREFIX ex: <http://example.org/twin/> "
    "SELECT ?o WHERE { "
    "  ?o a saref:Observation . "
    "  FILTER NOT EXISTS { ?o ex:observedAgent ?a } "
    "}"
)


def geofence_query(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> str:
    """Return a SELECT query for predicted points OUTSIDE the given lat/lon box.

    The returned query string is safe for ``sparql()`` (it is a SELECT).

    Parameters
    ----------
    lat_min, lat_max:
        WGS84 latitude bounds (decimal degrees).
    lon_min, lon_max:
        WGS84 longitude bounds (decimal degrees).

    Returns
    -------
    str
        SPARQL SELECT query string.
    """
    return (
        "PREFIX ex: <http://example.org/twin/> "
        "PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> "
        "PREFIX saref: <https://saref.etsi.org/core/> "
        "SELECT ?o ?lat ?lon WHERE { "
        '  ?o a saref:Observation ; ex:kind "predicted" ; '
        "     geo:lat ?lat ; geo:long ?lon . "
        f"  FILTER (?lat < {lat_min!r} || ?lat > {lat_max!r} || "
        f"          ?lon < {lon_min!r} || ?lon > {lon_max!r}) "
        "}"
    )


# ---------------------------------------------------------------------------
# Allowed algebra forms
# ---------------------------------------------------------------------------

_ALLOWED_ALGEBRA_NAMES = frozenset({"SelectQuery", "AskQuery"})


def _check_read_only(query_string: str) -> None:
    """Validate that *query_string* is a read-only SELECT or ASK.

    Raises
    ------
    QueryRejectedError
        If the query is a SPARQL Update, CONSTRUCT, DESCRIBE, or any other
        non-SELECT/ASK form.
    """
    try:
        parsed = prepareQuery(query_string)
    except Exception as exc:
        # prepareQuery raises ParseException for SPARQL Update forms
        # (INSERT, DELETE, etc.) as they are not valid query syntax.
        raise QueryRejectedError(
            f"Query rejected (parse error — likely a SPARQL Update form): {exc}"
        ) from exc

    algebra_name = parsed.algebra.name
    if algebra_name not in _ALLOWED_ALGEBRA_NAMES:
        raise QueryRejectedError(
            f"Query rejected: only SELECT and ASK are permitted; "
            f"got algebra type '{algebra_name}'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sparql(graph: rdflib.Graph, query_string: str) -> list[dict[str, Any]]:
    """Execute a read-only SPARQL query against *graph*.

    Parameters
    ----------
    graph:
        The rdflib.Graph to query.  This function NEVER mutates it.
    query_string:
        A SPARQL SELECT or ASK query.

    Returns
    -------
    list[dict]
        For SELECT: one dict per result row; values are ``.toPython()`` natives
        where available, otherwise the raw rdflib term.
        For ASK: ``[{"_ask": True}]`` or ``[{"_ask": False}]``.

    Raises
    ------
    QueryRejectedError
        If the query is not a SELECT or ASK (including all SPARQL Update forms).
    """
    _check_read_only(query_string)

    result = graph.query(query_string)

    if result.type == "ASK":
        return [{"_ask": bool(result.askAnswer)}]

    # SELECT
    rows: list[dict[str, Any]] = []
    for row in result:
        d = {}
        for var, val in row.asdict().items():
            if hasattr(val, "toPython"):
                d[var] = val.toPython()
            else:
                d[var] = val
        rows.append(d)
    return rows
