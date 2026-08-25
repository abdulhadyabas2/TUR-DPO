"""Regression tests for publication-facing annotation and peer-pressure paths."""

from turdpo.topology import (
    Edge,
    Node,
    TopologyGraph,
    TopologyScorer,
    parse_topology_output,
)


def _chain(peer=False):
    graph = TopologyGraph()
    graph.add_node(Node("n0", "premise", node_type="premise", is_peer_assertion=peer))
    graph.add_node(Node("n1", "derived", node_type="intermediate", is_peer_assertion=peer))
    graph.add_node(Node("n2", "answer", node_type="conclusion"))
    graph.add_edge(Edge("n0", "n1", edge_type="peer_assertion" if peer else "supports"))
    graph.add_edge(Edge("n1", "n2", edge_type="supports"))
    return graph


def test_json_parser_handles_nested_fenced_output_and_string_booleans():
    output = '''prefix
```json
{"nodes":[{"id":"n0","content":"premise","metadata":{"source":"peer"},"is_peer_assertion":"false"},
{"id":"n1","content":"answer","node_type":"conclusion"}],"edges":[{"source_id":"n0","target_id":"n1"}]}
```
suffix'''
    graph = parse_topology_output(output)
    assert len(graph.nodes) == 2
    assert graph.nodes["n0"].is_peer_assertion is False
    assert graph.nodes["n0"].metadata["source"] == "peer"


def test_peer_penalty_only_counts_minimal_path_reliance():
    independent = _chain(peer=False)
    pressured = _chain(peer=True)
    assert independent.peer_pressure_rate() == 0.0
    assert pressured.peer_pressure_rate() > 0.0
    scorer = TopologyScorer(alpha_peer_unverified=1.0)
    assert scorer.compute_score(pressured) < scorer.compute_score(independent)
