from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import re

Graph = Dict[str, Any]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_graph(text: str, max_nodes: int = 6) -> Graph:
    sents = split_sentences(text)
    nodes = sents[:max_nodes]
    n = len(nodes)
    edges: List[Tuple[int, int]] = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    # simple back-reference heuristic
    for j, s in enumerate(nodes):
        if re.search(r"\bstep\s*\d+|previous|earlier|above\b", s, flags=re.I):
            if j > 0:
                edges.append((j - 1, j))
    contra = 1.0 if re.search(r"\b(however|contradict|but|on the contrary)\b", " ".join(nodes), re.I) else 0.0
    return {"nodes": nodes, "edges": edges, "meta": {"contradiction": contra}}


def path_coverage(G: Graph) -> float:
    nodes: List[str] = G.get("nodes", [])
    edges: List[Tuple[int, int]] = G.get("edges", [])
    n = len(nodes)
    if n == 0:
        return 0.0
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        if 0 <= i < n and 0 <= j < n:
            adj[i].append(j)
    target = n - 1
    reachable = set()

    def dfs(u: int, seen: Tuple[int, ...]):
        if u in seen:
            return
        seen2 = seen + (u,)
        if u == target:
            reachable.update(seen2)
            return
        for v in adj.get(u, []):
            dfs(v, seen2)

    for i in range(n):
        dfs(i, tuple())
    return len(reachable) / float(n)


def cycle_count(G: Graph) -> float:
    nodes: List[str] = G.get("nodes", [])
    edges: List[Tuple[int, int]] = G.get("edges", [])
    n = len(nodes)
    back_edges = sum(1 for i, j in edges if j < i)
    two_cycles = sum(1 for i, j in edges if (j, i) in set(edges))
    return (back_edges + two_cycles) / max(1, n - 1)


def dangling(G: Graph) -> float:
    nodes: List[str] = G.get("nodes", [])
    edges: List[Tuple[int, int]] = G.get("edges", [])
    n = len(nodes)
    indeg = [0] * n
    outdeg = [0] * n
    for i, j in edges:
        if 0 <= i < n and 0 <= j < n:
            outdeg[i] += 1
            indeg[j] += 1
    d = 0
    for i in range(n):
        if i > 0 and indeg[i] == 0:
            d += 1
        if i < n - 1 and outdeg[i] == 0:
            d += 1
    return d / max(1, n)


def topology_score(G: Graph, alpha=(1.0, 1.0, 0.5, 0.7)) -> Dict[str, float]:
    pc = path_coverage(G)
    cy = cycle_count(G)
    dn = dangling(G)
    ct = float(G.get("meta", {}).get("contradiction", 0.0))
    a1, a2, a3, a4 = alpha
    s = a1 * pc - a2 * cy - a3 * dn - a4 * ct
    return {"path_cover": pc, "cycles": cy, "dangling": dn, "contrad": ct, "s_topo": s}


def aleatoric_uncertainty(G: Graph, tau: float = 0.05) -> float:
    nodes: List[str] = G.get("nodes", [])
    if not nodes:
        return 0.0
    ent = 0.0
    for s in nodes:
        has_num = bool(re.search(r"\d", s))
        has_cue = bool(re.search(r"\b(therefore|thus|so|hence)\b", s, flags=re.I))
        p = 0.65 + 0.2 * has_num + 0.1 * has_cue
        p = min(0.95, max(0.05, p))
        p_t = (p + tau) / (1 + 2 * tau)
        p_t = min(1 - 1e-8, max(1e-8, p_t))
        ent += -(p_t * math.log(p_t) + (1 - p_t) * math.log(1 - p_t))
    return ent / len(nodes)


def epistemic_uncertainty(graphs: List[Graph]) -> float:
    if not graphs:
        return 0.0
    scores = [topology_score(g)["s_topo"] for g in graphs]
    mu = sum(scores) / len(scores)
    var = sum((s - mu) ** 2 for s in scores) / max(1, len(scores) - 1)
    # simple structure dispersion: std of node counts
    ns = [len(g.get("nodes", [])) for g in graphs]
    mu_n = sum(ns) / len(ns)
    var_n = sum((n - mu_n) ** 2 for n in ns) / max(1, len(ns) - 1)
    return float(var + 0.1 * var_n)
