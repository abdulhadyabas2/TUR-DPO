"""
Topology Module for TUR-DPO

This module implements topology extraction and scoring for reasoning graphs.
Based on Equation (1) from the paper:
    s_topo(G) = α₁ q_path - α₂ c_cycle - α₃ d_dangling - α₄ q_contradict
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from collections import deque
import re
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Represents an atomic subclaim or reasoning step in the topology graph."""
    id: str
    content: str
    node_type: str = "claim"  # "claim", "premise", "conclusion", "intermediate"
    correctness_prob: float = 0.5
    is_peer_assertion: bool = False  # Set to True for peer-derived claims (multi-agent)
    peer_verified: bool = False      # Set to True if peer claim has external/self verification
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False


@dataclass
class Edge:
    """Represents a support or dependency relation between nodes."""
    source_id: str
    target_id: str
    edge_type: str = "supports"  # "supports", "contradicts", "depends", "peer_assertion"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.source_id, self.target_id))


class TopologyGraph:
    """
    Directed graph representing the reasoning topology of a response.

    Nodes represent atomic subclaims/steps, edges represent support relations.
    Provides methods for structural analysis (cycles, paths, dangling nodes).
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[str]] = {}  # outgoing edges
        self.reverse_adjacency: Dict[str, List[str]] = {}  # incoming edges

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
        if node.id not in self.reverse_adjacency:
            self.reverse_adjacency[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        # Ensure nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"Both nodes must exist before adding edge: {edge.source_id} -> {edge.target_id}")

        # Prevent self-loops
        if edge.source_id == edge.target_id:
            return

        self.edges.append(edge)
        self.adjacency[edge.source_id].append(edge.target_id)
        self.reverse_adjacency[edge.target_id].append(edge.source_id)

    def get_premises(self) -> List[Node]:
        """Get nodes with no incoming edges (premises/starting points)."""
        return [node for node_id, node in self.nodes.items()
                if len(self.reverse_adjacency.get(node_id, [])) == 0]

    def get_conclusions(self) -> List[Node]:
        """Get nodes with no outgoing edges (conclusions/final claims)."""
        return [node for node_id, node in self.nodes.items()
                if len(self.adjacency.get(node_id, [])) == 0]

    def get_dangling_nodes(self) -> List[Node]:
        """
        Get dangling nodes - nodes that are not part of any connected reasoning path
        from a premise to a conclusion.
        """
        if len(self.nodes) <= 1:
            return []

        # In a graph with multiple nodes, an isolated node (degree 0) is dangling
        isolated_nodes = {
            node_id for node_id in self.nodes
            if len(self.adjacency.get(node_id, [])) == 0 and len(self.reverse_adjacency.get(node_id, [])) == 0
        }

        # Non-isolated premises (must have outgoing edges)
        active_premises = [
            node for node in self.get_premises()
            if len(self.adjacency.get(node.id, [])) > 0
        ]

        # Non-isolated conclusions (must have incoming edges)
        active_conclusions = [
            node for node in self.get_conclusions()
            if len(self.reverse_adjacency.get(node.id, [])) > 0
        ]

        reachable_from_premises = set()
        for premise in active_premises:
            reachable_from_premises.update(self._bfs_reachable(premise.id, forward=True))

        can_reach_conclusions = set()
        for conclusion in active_conclusions:
            can_reach_conclusions.update(self._bfs_reachable(conclusion.id, forward=False))

        # Valid path nodes must be reachable from an active premise and reach an active conclusion
        valid_path_nodes = reachable_from_premises & can_reach_conclusions

        dangling = [
            node for node_id, node in self.nodes.items()
            if node_id in isolated_nodes or node_id not in valid_path_nodes
        ]

        return dangling

    def get_unverified_peer_nodes(self) -> List[Node]:
        """Get all nodes explicitly marked as unverified peer assertions."""
        return [
            node for node in self.nodes.values()
            if getattr(node, 'is_peer_assertion', False) and not getattr(node, 'peer_verified', False)
        ]

    def get_peer_pressure_nodes(self) -> List[Node]:
        """Return minimal-path nodes whose only incoming support is unverified peer support.

        Premises are excluded because they do not have incoming support.  A node
        qualifies when every incoming support edge originates at an unverified peer
        assertion.  This makes the extension precise and prevents unrelated peer
        mentions elsewhere in a response from receiving a penalty.
        """
        path_node_ids = {
            node_id
            for path in self.get_minimal_valid_paths()
            for node_id in path
        }
        peer_nodes = []
        for node_id in path_node_ids:
            incoming = [
                edge for edge in self.edges
                if edge.target_id == node_id
                and edge.edge_type in {"supports", "depends", "peer_assertion"}
            ]
            if not incoming:
                continue
            only_peer_support = all(
                self.nodes[edge.source_id].is_peer_assertion
                and not self.nodes[edge.source_id].peer_verified
                for edge in incoming
            )
            if only_peer_support:
                peer_nodes.append(self.nodes[node_id])
        return peer_nodes

    def peer_pressure_rate(self) -> float:
        """Return the fraction of minimal-path nodes affected by peer pressure."""
        path_node_ids = {
            node_id
            for path in self.get_minimal_valid_paths()
            for node_id in path
        }
        if not path_node_ids:
            return 0.0
        return len(self.get_peer_pressure_nodes()) / len(path_node_ids)

    def _bfs_reachable(self, start_id: str, forward: bool = True) -> Set[str]:
        """BFS to find all reachable nodes from start_id."""
        visited = set()
        queue = deque([start_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            neighbors = self.adjacency.get(current, []) if forward else self.reverse_adjacency.get(current, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        return visited

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the graph using DFS.
        Returns list of cycles found.
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in self.adjacency.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)

        return cycles

    def count_cycles(self) -> int:
        """Count the number of cycles in the graph."""
        return len(self.detect_cycles())

    def get_minimal_valid_paths(self) -> List[List[str]]:
        """
        Find all minimal valid paths from premises to conclusions.
        A valid path connects a premise to a conclusion.
        """
        premises = self.get_premises()
        conclusions = self.get_conclusions()
        all_paths = []

        for premise in premises:
            for conclusion in conclusions:
                paths = self._find_all_paths(premise.id, conclusion.id)
                all_paths.extend(paths)

        return all_paths

    def _find_all_paths(self, start_id: str, end_id: str, max_depth: int = 20) -> List[List[str]]:
        """Find all paths from start to end using DFS with depth limit."""
        paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return
            if current == end_id:
                paths.append(path.copy())
                return

            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        dfs(start_id, [start_id], {start_id})
        return paths

    def compute_path_coverage(self) -> float:
        """
        Compute the fraction of nodes that participate in at least one
        valid path from premises to conclusions.
        """
        if len(self.nodes) == 0:
            return 0.0

        paths = self.get_minimal_valid_paths()
        nodes_in_paths = set()
        for path in paths:
            nodes_in_paths.update(path)

        return len(nodes_in_paths) / len(self.nodes)

    def compute_edge_path_coverage(self) -> float:
        """Compute the fraction of edges that lie on at least one valid path."""
        if not self.edges:
            return 0.0
        path_edges = {
            (path[index], path[index + 1])
            for path in self.get_minimal_valid_paths()
            for index in range(len(path) - 1)
        }
        return sum(
            1 for edge in self.edges if (edge.source_id, edge.target_id) in path_edges
        ) / len(self.edges)

    def get_edge_distribution(self) -> Dict[str, float]:
        """Get normalized distribution over edges for JSD computation."""
        if len(self.edges) == 0:
            return {}

        dist = {}
        for edge in self.edges:
            key = f"{edge.source_id}->{edge.target_id}"
            dist[key] = dist.get(key, 0) + edge.weight

        # Normalize
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}

    def get_path_distribution(self) -> Dict[str, float]:
        """Get normalized distribution over paths for JSD computation."""
        paths = self.get_minimal_valid_paths()
        if not paths:
            return {}

        dist = {}
        for path in paths:
            key = "->".join(path)
            dist[key] = dist.get(key, 0) + 1

        # Normalize
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}

    def sanitize(self) -> 'TopologyGraph':
        """
        Sanitize the graph by:
        1. Removing self-loops
        2. Breaking cycles by minimal edge cut
        3. Merging paraphrase nodes (if similarity > threshold)
        """
        self._merge_near_duplicate_nodes()

        # Remove existing cycles by removing back edges
        cycles = self.detect_cycles()
        edges_to_remove = set()

        for cycle in cycles:
            if len(cycle) > 1:
                # Remove the edge that creates the cycle (last edge)
                edges_to_remove.add((cycle[-2], cycle[-1]))

        # Filter out problematic edges
        self.edges = [e for e in self.edges
                     if (e.source_id, e.target_id) not in edges_to_remove]

        # Rebuild adjacency lists
        self.adjacency = {node_id: [] for node_id in self.nodes}
        self.reverse_adjacency = {node_id: [] for node_id in self.nodes}

        for edge in self.edges:
            self.adjacency[edge.source_id].append(edge.target_id)
            self.reverse_adjacency[edge.target_id].append(edge.source_id)

        return self

    def _merge_near_duplicate_nodes(self, threshold: float = 0.92) -> None:
        """Merge exact or near-duplicate node text and rebuild graph indexes."""
        node_ids = list(self.nodes)
        replacement = {node_id: node_id for node_id in node_ids}
        for index, left_id in enumerate(node_ids):
            left = re.sub(r"\s+", " ", self.nodes[left_id].content.strip().lower())
            for right_id in node_ids[index + 1:]:
                if replacement[right_id] != right_id:
                    continue
                right = re.sub(r"\s+", " ", self.nodes[right_id].content.strip().lower())
                if SequenceMatcher(None, left, right).ratio() >= threshold:
                    replacement[right_id] = left_id
        if all(key == value for key, value in replacement.items()):
            return

        self.nodes = {
            node_id: node for node_id, node in self.nodes.items()
            if replacement[node_id] == node_id
        }
        merged_edges = []
        seen_edges = set()
        for edge in self.edges:
            source_id = replacement[edge.source_id]
            target_id = replacement[edge.target_id]
            if source_id == target_id:
                continue
            edge_key = (source_id, target_id, edge.edge_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            merged_edges.append(Edge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                metadata=edge.metadata,
            ))
        self.edges = merged_edges
        self.adjacency = {node_id: [] for node_id in self.nodes}
        self.reverse_adjacency = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            self.adjacency[edge.source_id].append(edge.target_id)
            self.reverse_adjacency[edge.target_id].append(edge.source_id)

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"TopologyGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


class TopologyExtractor:
    """
    Extract reasoning topology from text responses.

    Decomposes text into atomic statements and links support relations.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        extraction_prompt_template: Optional[str] = None,
        max_nodes: int = 10,
        max_edges: int = 20
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self.extraction_prompt_template = extraction_prompt_template or self._default_prompt()

    def _default_prompt(self) -> str:
        # The JSON schema is declared below at module load time.  Looking it up
        # when an extractor is instantiated keeps the default and callable-backed
        # extractors on exactly the same output contract.
        return TOPOLOGY_ELICITATION_PROMPT_TEMPLATE

    def extract(
        self,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        perturbation: bool = False
    ) -> TopologyGraph:
        """
        Extract topology graph from a response.

        Args:
            prompt: The input prompt
            response: The model's response
            temperature: Sampling temperature (for perturbation)
            perturbation: Whether to add prompt perturbations for uncertainty estimation

        Returns:
            TopologyGraph representing the reasoning structure
        """
        if self.model is None:
            # Fallback to rule-based extraction
            return self._rule_based_extract(response)

        # Use model-based extraction
        extraction_prompt = self.extraction_prompt_template.format(
            prompt=prompt,
            response=response
        )

        if perturbation:
            # Add minor variations for epistemic uncertainty estimation
            extraction_prompt = self._add_perturbation(extraction_prompt)

        if self.tokenizer is None or not hasattr(self.model, "generate"):
            return self._rule_based_extract(response)

        try:
            import torch

            encoded = self.tokenizer(
                extraction_prompt,
                return_tensors="pt",
                truncation=True,
            )
            model_device = next(self.model.parameters()).device
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            generation_kwargs = {
                "max_new_tokens": 512,
                "do_sample": temperature > 0,
                "temperature": max(temperature, 1e-5),
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            }
            generation_kwargs = {
                key: value for key, value in generation_kwargs.items() if value is not None
            }
            with torch.no_grad():
                generated = self.model.generate(**encoded, **generation_kwargs)
            prompt_length = encoded["input_ids"].shape[-1]
            output = self.tokenizer.decode(
                generated[0][prompt_length:], skip_special_tokens=True
            )
            return parse_topology_output(output, max_nodes=self.max_nodes, max_edges=self.max_edges)
        except Exception as exc:
            logger.warning("Model topology extraction failed (%s); using rule-based extraction", exc)
            return self._rule_based_extract(response)

    def _rule_based_extract(self, response: str) -> TopologyGraph:
        """
        Rule-based topology extraction using sentence segmentation.
        """
        graph = TopologyGraph()

        # Split into sentences
        sentences = self._split_sentences(response)

        if not sentences:
            return graph

        # Create nodes for each sentence
        for i, sentence in enumerate(sentences[:self.max_nodes]):
            node_type = "premise" if i == 0 else ("conclusion" if i == len(sentences) - 1 else "intermediate")
            is_peer_assertion, peer_verified = self._infer_peer_flags(sentence)
            node = Node(
                id=f"n{i}",
                content=sentence.strip(),
                node_type=node_type,
                is_peer_assertion=is_peer_assertion,
                peer_verified=peer_verified,
            )
            graph.add_node(node)

        # Create sequential edges (simple chain structure)
        for i in range(min(len(sentences) - 1, self.max_nodes - 1)):
            edge = Edge(
                source_id=f"n{i}",
                target_id=f"n{i+1}",
                edge_type="supports"
            )
            graph.add_edge(edge)

        return graph.sanitize()

    def _infer_peer_flags(self, sentence: str) -> Tuple[bool, bool]:
        """Apply conservative peer-assertion flags for the rule-based fallback."""
        lowered = sentence.lower()
        peer_marker = re.search(
            r"\b(agent|peer|peers|classmate|colleague|everyone|they)\b.{0,80}\b"
            r"(said|says|claimed|asserted|agreed|believes|believe|thinks|think)\b",
            lowered,
        )
        if peer_marker is None and not re.search(r"\b(my peers|according to the other agent)\b", lowered):
            return False, False
        verified = bool(re.search(r"\b(verified|verify|checked|confirmed|proof|calculate|derived)\b", lowered))
        return True, verified

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _add_perturbation(self, prompt: str) -> str:
        """Add minor perturbations to prompt for uncertainty estimation."""
        perturbations = [
            "Please analyze carefully: ",
            "Think step by step: ",
            "Consider the reasoning: ",
            "",
        ]
        import random
        return random.choice(perturbations) + prompt

    def extract_multiple(
        self,
        prompt: str,
        response: str,
        k: int = 3
    ) -> List[TopologyGraph]:
        """
        Extract K topology graphs with perturbations for epistemic uncertainty.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        graphs = []
        for i in range(k):
            graph = self.extract(
                prompt=prompt,
                response=response,
                temperature=0.3 if i > 0 else 0.0,
                perturbation=(i > 0)
            )
            graphs.append(graph)
        return graphs


class TopologyScorer:
    """
    Compute topology score from graph features.

    Based on Equation (1):
        s_topo(G) = α₁ q_path - α₂ c_cycle - α₃ d_dangling - α₄ q_contradict - α_peer p_peer
    """

    def __init__(
        self,
        alpha_path: float = 1.0,
        alpha_cycle: float = 0.5,
        alpha_dangling: float = 0.3,
        alpha_contradict: float = 0.4,
        alpha_peer_unverified: float = 0.0,
        normalize: bool = True,
        score_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize topology scorer.

        Args:
            alpha_path: Weight for path coverage (positive contribution)
            alpha_cycle: Weight for cycle count (negative contribution)
            alpha_dangling: Weight for dangling nodes (negative contribution)
            alpha_contradict: Weight for contradiction score (negative contribution)
            alpha_peer_unverified: Weight for unverified peer assertions penalty (multi-agent extension)
            normalize: Whether to normalize score to [0, 1]
            score_range: Target range for normalized scores
        """
        self.alpha_path = alpha_path
        self.alpha_cycle = alpha_cycle
        self.alpha_dangling = alpha_dangling
        self.alpha_contradict = alpha_contradict
        self.alpha_peer_unverified = alpha_peer_unverified
        self.normalize = normalize
        self.score_range = score_range

    def compute_score(
        self,
        graph: TopologyGraph,
        contradiction_score: float = 0.0,
        peer_penalty: Optional[float] = None
    ) -> float:
        """
        Compute topology score for a graph.

        Args:
            graph: The topology graph to score
            contradiction_score: Pre-computed contradiction score (from NLI verifier)
            peer_penalty: Optional pre-computed unverified peer penalty. If None,
                          it is computed from unverified peer nodes in graph.

        Returns:
            Topology score (higher is better)
        """
        if len(graph) == 0:
            return 0.0

        # Compute bounded components so normalization remains meaningful for
        # malformed or highly cyclic elicited graphs.
        node_path_coverage = graph.compute_path_coverage()
        edge_path_coverage = graph.compute_edge_path_coverage()
        q_path = float(np.clip(
            (node_path_coverage + edge_path_coverage) / 2.0 if graph.edges else node_path_coverage,
            0.0,
            1.0,
        ))
        c_cycle = float(np.clip(graph.count_cycles() / max(len(graph), 1), 0.0, 1.0))
        d_dangling = float(np.clip(len(graph.get_dangling_nodes()) / len(graph), 0.0, 1.0))
        q_contradict = float(np.clip(contradiction_score, 0.0, 1.0))

        if peer_penalty is None:
            p_peer = graph.peer_pressure_rate()
        else:
            p_peer = float(np.clip(peer_penalty, 0.0, 1.0))

        # Apply Equation (1) + Multi-Agent Peer Penalty extension
        score = (
            self.alpha_path * q_path
            - self.alpha_cycle * c_cycle
            - self.alpha_dangling * d_dangling
            - self.alpha_contradict * q_contradict
            - self.alpha_peer_unverified * p_peer
        )

        if self.normalize:
            # Normalize to score_range
            min_score = -self.alpha_cycle - self.alpha_dangling - self.alpha_contradict - self.alpha_peer_unverified
            max_score = self.alpha_path

            if max_score > min_score:
                normalized = (score - min_score) / (max_score - min_score)
                score = self.score_range[0] + normalized * (self.score_range[1] - self.score_range[0])
            else:
                score = (self.score_range[0] + self.score_range[1]) / 2

        return float(np.clip(score, min(self.score_range), max(self.score_range))) if self.normalize else float(score)

    def compute_features(self, graph: TopologyGraph) -> Dict[str, float]:
        """Compute all topology features for analysis."""
        return {
            "path_coverage": graph.compute_path_coverage(),
            "edge_path_coverage": graph.compute_edge_path_coverage(),
            "cycle_count": graph.count_cycles(),
            "dangling_count": len(graph.get_dangling_nodes()),
            "unverified_peer_count": len(graph.get_unverified_peer_nodes()),
            "peer_pressure_count": len(graph.get_peer_pressure_nodes()),
            "peer_pressure_rate": graph.peer_pressure_rate(),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "num_premises": len(graph.get_premises()),
            "num_conclusions": len(graph.get_conclusions()),
        }


TOPOLOGY_ELICITATION_PROMPT_TEMPLATE = """You are a reasoning graph extractor. Given a question and a candidate response, break down the response into an atomic reasoning graph in JSON format.

Nodes represent individual atomic claims, premises, or intermediate derivation steps.
Edges represent inferential support or dependency relations between steps.

Output strictly valid JSON with this format:
```json
{{
  "nodes": [
    {{
      "id": "n0",
      "content": "Given premise or starting claim",
      "node_type": "premise",
      "is_peer_assertion": false,
      "peer_verified": false
    }},
    {{
      "id": "n1",
      "content": "Intermediate deduction step",
      "node_type": "intermediate",
      "is_peer_assertion": false,
      "peer_verified": false
    }},
    {{
      "id": "n2",
      "content": "Final conclusion",
      "node_type": "conclusion",
      "is_peer_assertion": false,
      "peer_verified": false
    }}
  ],
  "edges": [
    {{
      "source_id": "n0",
      "target_id": "n1",
      "edge_type": "supports"
    }},
    {{
      "source_id": "n1",
      "target_id": "n2",
      "edge_type": "supports"
    }}
  ]
}}
```

Question:
{prompt}

Candidate Response:
{response}

JSON:"""


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce JSON booleans and common string spellings safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    if value is None:
        return default
    return bool(value)


def _extract_json_object(raw_output: str) -> Dict[str, Any]:
    """Extract the first decodable JSON object from fenced or conversational output."""
    import json

    if not isinstance(raw_output, str):
        raise TypeError("Topology elicitation output must be a string")
    text = raw_output.strip()
    fenced = re.findall(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
    )
    decoder = json.JSONDecoder()
    for candidate in fenced + [text]:
        candidate = candidate.strip()
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("No valid JSON object found in topology elicitation output")


def parse_topology_output(
    raw_output: str,
    max_nodes: int = 15,
    max_edges: int = 30,
) -> TopologyGraph:
    """Parse and validate an elicited topology graph.

    The parser accepts raw JSON or a JSON object inside a Markdown code fence.
    Invalid node references, duplicate ids, malformed values, and extra fields
    are handled deterministically.  A graph with no valid nodes is rejected so
    callers can use the documented fallback extractor.
    """
    data = _extract_json_object(raw_output)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Topology JSON must contain list-valued 'nodes' and 'edges'")

    graph = TopologyGraph()
    used_ids = set()
    for index, node_data in enumerate(nodes[:max_nodes]):
        if not isinstance(node_data, dict):
            continue
        node_id = str(node_data.get("id", f"n{index}")).strip() or f"n{index}"
        if node_id in used_ids:
            node_id = f"{node_id}_{index}"
        content = str(node_data.get("content", "")).strip()
        if not content:
            continue
        try:
            correctness_prob = float(node_data.get("correctness_prob", 0.5))
        except (TypeError, ValueError):
            correctness_prob = 0.5
        graph.add_node(Node(
            id=node_id,
            content=content,
            node_type=str(node_data.get("node_type", "claim")),
            correctness_prob=float(np.clip(correctness_prob, 0.0, 1.0)),
            is_peer_assertion=_coerce_bool(node_data.get("is_peer_assertion")),
            peer_verified=_coerce_bool(node_data.get("peer_verified")),
            metadata=node_data.get("metadata") if isinstance(node_data.get("metadata"), dict) else {},
        ))
        used_ids.add(node_id)

    if not graph.nodes:
        raise ValueError("Topology JSON did not contain any valid nodes")

    for edge_data in edges[:max_edges]:
        if not isinstance(edge_data, dict):
            continue
        source_id = str(edge_data.get("source_id", ""))
        target_id = str(edge_data.get("target_id", ""))
        if source_id not in graph.nodes or target_id not in graph.nodes:
            continue
        try:
            weight = float(edge_data.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        graph.add_edge(Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=str(edge_data.get("edge_type", "supports")),
            weight=weight,
            metadata=edge_data.get("metadata") if isinstance(edge_data.get("metadata"), dict) else {},
        ))
    return graph.sanitize()


class LLMTopologyExtractor(TopologyExtractor):
    """
    LLM-based reasoning topology extractor using prompt templates.
    Parses structured JSON graphs output by an LLM, with fallback to rule-based parsing.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        prompt_template: str = TOPOLOGY_ELICITATION_PROMPT_TEMPLATE,
        max_nodes: int = 15,
        **kwargs
    ):
        super().__init__(max_nodes=max_nodes, **kwargs)
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template

    def extract_with_llm(
        self,
        prompt: str,
        response: str,
        llm_fn: Optional[Callable[[str], str]] = None
    ) -> TopologyGraph:
        """
        Extract topology graph using LLM call.
        """
        fn = llm_fn or self.llm_fn
        if fn is None:
            # Fallback to rule-based if no LLM callable provided
            return super().extract(prompt, response)

        query_prompt = self.prompt_template.format(prompt=prompt, response=response)
        raw_output = fn(query_prompt)

        try:
            return parse_topology_output(
                raw_output,
                max_nodes=self.max_nodes,
                max_edges=self.max_edges,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse LLM topology output (%s); using rule-based extraction",
                exc,
            )
            return super().extract(prompt, response)

    def extract(
        self,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        perturbation: bool = False,
    ) -> TopologyGraph:
        """Use the configured callable; otherwise use the base extractor."""
        if self.llm_fn is None:
            return super().extract(prompt, response, temperature, perturbation)
        return self.extract_with_llm(prompt, response)
