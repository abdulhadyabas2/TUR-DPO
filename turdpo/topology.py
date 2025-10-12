from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import re
from collections import Counter

Graph = Dict[str, Any]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    # Simple sentence splitter; fall back to line/period splitting
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_topology(response: str, max_nodes: int = 6) -> Graph:
    """
    Heuristic topology extractor:
    - Nodes: up to first `max_nodes` sentences/clauses.
    - Edges: chain edges (i -> i+1). If a sentence references a previous step keyword, add a back edge.
    - Flags basic contradictions via cue words.
    Returns a graph dict: {nodes: [str], edges: [(i,j)], meta: {...}}
    """
    sents = _split_sentences(response)
    if not sents:
        return {"nodes": [], "edges": [], "meta": {"contradiction": 0.0}}

    nodes = sents[:max_nodes]
    n = len(nodes)
    edges: List[Tuple[int, int]] = []

    # chain edges
    for i in range(n - 1):
        edges.append((i, i + 1))

    # back-references: if sentence mentions step keywords
    step_pattern = re.compile(r"(step\s*(\d+)|previous|earlier|above)", re.I)
    for j, s in enumerate(nodes):
        for m in re.finditer(step_pattern, s):
            # add an edge from mentioned step (if any) to j
            # crude heuristic: if 'step d' use d-1, else connect j-1
            g = m.group(2)
            if g:
                idx = max(0, min(n - 1, int(g) - 1))
                if idx != j:
                    edges.append((idx, j))
            elif j > 0:
                edges.append((j - 1, j))

    # contradiction cues
    contra_cues = [
        r"however",
        r"but",
        r"nevertheless",
        r"contradict",
        r"on the contrary",
        r"yet",
        r"nonetheless",
    ]
    contra = 0
    for s in nodes:
        if re.search(r"|".join(contra_cues), s, flags=re.I):
            contra += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "contradiction": float(contra > 0),
        },
    }


def _acyclic_path_coverage(nodes: List[str], edges: List[Tuple[int, int]]) -> float:
    """Fraction of nodes that belong to at least one acyclic path ending at the last node."""
    n = len(nodes)
    if n == 0:
        return 0.0
    # Build adjacency
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in edges:
        if 0 <= i < n and 0 <= j < n:
            adj[i].append(j)

    # simple DFS to collect nodes reaching final node (n-1) without revisits
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


def _cycle_count(n: int, edges: List[Tuple[int, int]]) -> int:
    # crude cycle detection: count edges that point backwards or form 2-cycles
    back_edges = sum(1 for i, j in edges if j < i)
    pair_edges = set(edges)
    two_cycles = sum(1 for i, j in edges if (j, i) in pair_edges)
    return back_edges + two_cycles


def _dangling_nodes(n: int, edges: List[Tuple[int, int]]) -> int:
    indeg = [0] * n
    outdeg = [0] * n
    for i, j in edges:
        if 0 <= i < n and 0 <= j < n:
            outdeg[i] += 1
            indeg[j] += 1
    # dangling: nodes with zero indeg (except first) or zero outdeg (except last)
    d = 0
    for i in range(n):
        if i > 0 and indeg[i] == 0:
            d += 1
        if i < n - 1 and outdeg[i] == 0:
            d += 1
    return d


def topology_score(G: Graph, alpha=(1.0, 1.0, 0.5, 0.7)) -> Dict[str, float]:
    """
    Compute s_topo per Eq. (topology components):
    s_topo = a1 * path_cover - a2 * cycles - a3 * dangling - a4 * contradiction
    Returns dict with components and total.
    """
    nodes: List[str] = G.get("nodes", [])
    edges: List[Tuple[int, int]] = G.get("edges", [])
    n = len(nodes)
    if n == 0:
        return {"path_cover": 0.0, "cycles": 0.0, "dangling": 0.0, "contrad": 0.0, "s_topo": 0.0}

    path_cover = _acyclic_path_coverage(nodes, edges)
    cycles = float(_cycle_count(n, edges)) / max(1, n - 1)  # normalize
    dangling = float(_dangling_nodes(n, edges)) / max(1, n)
    contrad = float(G.get("meta", {}).get("contradiction", 0.0))

    a1, a2, a3, a4 = alpha
    s = a1 * path_cover - a2 * cycles - a3 * dangling - a4 * contrad
    return {
        "path_cover": path_cover,
        "cycles": cycles,
        "dangling": dangling,
        "contrad": contrad,
        "s_topo": s,
    }


def _hist_distribution(values: List[int], bins: List[int]) -> List[float]:
    c = Counter()
    for v in values:
        placed = False
        for b in bins:
            if v <= b:
                c[b] += 1
                placed = True
                break
        if not placed:
            c[float("inf")] += 1
    total = sum(c.values()) or 1
    return [c[b] / total for b in bins + [float("inf")]]


def _jsd(p: List[float], q: List[float]) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def _kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                s += ai * math.log(ai / bi)
        return s
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def epistemic_uncertainty(graphs: List[Graph]) -> float:
    """Variance of s_topo across samples + JSD over out-degree histograms as a simple proxy."""
    if not graphs:
        return 0.0
    scores = [topology_score(g)["s_topo"] for g in graphs]
    mu = sum(scores) / len(scores)
    var = sum((s - mu) ** 2 for s in scores) / max(1, len(scores) - 1)

    # degree histogram divergence across first two graphs (fallback average over pairs)
    def out_hist(g: Graph) -> List[float]:
        n = len(g.get("nodes", []))
        outdeg = [0] * n
        for i, j in g.get("edges", []):
            if 0 <= i < n and 0 <= j < n:
                outdeg[i] += 1
        return _hist_distribution(outdeg, bins=[0, 1, 2])

    if len(graphs) == 1:
        jsd = 0.0
    else:
        jsds = []
        base = out_hist(graphs[0])
        for g in graphs[1:]:
            jsds.append(_jsd(base, out_hist(g)))
        jsd = sum(jsds) / len(jsds)

    return float(var + jsd)


def aleatoric_uncertainty(G: Graph, node_probs: List[float] | None = None, tau: float = 0.05) -> float:
    """
    Average coverage-corrected binary entropy across nodes.
    If node_probs is None, estimate a crude probability from punctuation consistency cues.
    """
    nodes: List[str] = G.get("nodes", [])
    if not nodes:
        return 0.0

    if node_probs is None:
        # crude proxy: sentences with numbers or explicit therefore/so treated as more reliable
        probs = []
        for s in nodes:
            has_num = bool(re.search(r"\d", s))
            has_cue = bool(re.search(r"\b(therefore|so|thus|hence)\b", s, flags=re.I))
            p = 0.65 + 0.2 * has_num + 0.1 * has_cue
            probs.append(min(0.95, max(0.05, p)))
        node_probs = probs

    ent = 0.0
    for p in node_probs:
        p_tilde = (p + tau) / (1.0 + 2.0 * tau)
        p_tilde = min(1 - 1e-8, max(1e-8, p_tilde))
        ent += -(p_tilde * math.log(p_tilde) + (1 - p_tilde) * math.log(1 - p_tilde))

    return float(ent / len(nodes))
