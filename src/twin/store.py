"""Persistence helpers for the semantic twin graph.

Serialises an rdflib.Graph to Turtle and loads it back.
The Turtle format is chosen for human readability and portability — it is a
text file that can be inspected, diffed, and read by any RDF toolchain.
"""
from __future__ import annotations

import pathlib

import rdflib


def save_turtle(graph: rdflib.Graph, path: pathlib.Path) -> None:
    """Serialise *graph* to Turtle format at *path*.

    Parameters
    ----------
    graph:
        The in-memory RDF graph to persist.
    path:
        Destination file path (will be created or overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(path), format="turtle")


def load_turtle(path: pathlib.Path) -> rdflib.Graph:
    """Load a Turtle file from *path* into a new in-memory rdflib.Graph.

    Parameters
    ----------
    path:
        Source Turtle file.

    Returns
    -------
    rdflib.Graph
        Populated graph; namespace bindings are preserved from the file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Turtle file not found: {path}")
    g = rdflib.Graph()
    g.parse(str(path), format="turtle")
    return g
