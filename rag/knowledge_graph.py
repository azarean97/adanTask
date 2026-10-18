"""
Standard Clinical Knowledge Graph Module.
Builds an in-memory graph representation of entities, disease stages, diagnostic markers,
and clinical actions extracted from guidelines and VLM triples.
Saves graph structures to the outputs directory.
"""

import json
from collections import defaultdict
from typing import List, Dict, Any
from config import OUTPUTS_DIR

class ClinicalKnowledgeGraph:
    """Standard in-memory relational Knowledge Graph."""
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self._initialize_standard_ontology()

    def add_node(self, node_id: str, label: str = None, node_type: str = "Concept", properties: Dict[str, Any] = None):
        """Add a clinical entity node."""
        node_id = str(node_id).strip()
        if not node_id:
            return
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "label": label or node_id,
                "type": node_type,
                "properties": properties or {}
            }

    def add_edge(self, source: str, relation: str, target: str, weight: float = 1.0):
        """Add a directed relational edge."""
        source = str(source).strip()
        target = str(target).strip()
        relation = str(relation).strip()
        if not source or not target:
            return

        self.add_node(source)
        self.add_node(target)

        for existing in self.adj[source]:
            if existing["target"] == target and existing["relation"] == relation:
                return

        self.adj[source].append({
            "target": target,
            "relation": relation,
            "weight": weight
        })

    def _initialize_standard_ontology(self):
        """Seed baseline CKD clinical concepts and relationships."""
        stages = [
            ("Stage 1", "Stage 1 (Normal/High GFR >= 90 with Kidney Damage)", "Stage"),
            ("Stage 2", "Stage 2 (Mild GFR 60-89 with Kidney Damage)", "Stage"),
            ("Stage 3", "Stage 3 (Moderate GFR 30-59)", "Stage"),
            ("Stage 4", "Stage 4 (Severe GFR 15-29)", "Stage"),
            ("Stage 5", "Stage 5 (Kidney Failure GFR < 15 / Dialysis)", "Stage"),
        ]
        for nid, lbl, ntype in stages:
            self.add_node(nid, lbl, ntype)

        biomarkers = [
            ("Serum Creatinine", "Serum Creatinine (sc)", "Biomarker"),
            ("eGFR", "Estimated Glomerular Filtration Rate", "Measure"),
            ("Albuminuria", "Albumin in Urine (al)", "DamageMarker"),
            ("Hypertension", "Hypertension (htn)", "Comorbidity"),
            ("Diabetes Mellitus", "Diabetes Mellitus (dm)", "Etiology"),
            ("Anemia", "Anemia / Low Hemoglobin (hemo)", "Complication"),
            ("Blood Urea", "Blood Urea (bu)", "Biomarker"),
        ]
        for nid, lbl, ntype in biomarkers:
            self.add_node(nid, lbl, ntype)

        actions = [
            ("CVD Risk Reduction", "Cardiovascular Disease Risk Reduction", "Action"),
            ("Slow Progression", "Slowing Progression of Kidney Disease", "Action"),
            ("Manage Complications", "Evaluating & Treating Complications", "Action"),
            ("Nephrology Referral", "Referral to Nephrologist", "Action"),
            ("Dialysis Preparation", "Vascular Access & Dialysis Preparation", "Action"),
            ("Kidney Replacement", "Kidney Replacement Therapy", "Action")
        ]
        for nid, lbl, ntype in actions:
            self.add_node(nid, lbl, ntype)

        edges = [
            ("Serum Creatinine", "calculates", "eGFR"),
            ("Albuminuria", "indicates", "Stage 1"),
            ("Albuminuria", "indicates", "Stage 2"),
            ("eGFR < 15", "defines", "Stage 5"),
            ("eGFR 15-29", "defines", "Stage 4"),
            ("eGFR 30-59", "defines", "Stage 3"),
            ("eGFR 60-89", "defines", "Stage 2"),
            ("eGFR >= 90", "defines", "Stage 1"),
            ("Stage 1", "progresses_to", "Stage 2"),
            ("Stage 2", "progresses_to", "Stage 3"),
            ("Stage 3", "progresses_to", "Stage 4"),
            ("Stage 4", "progresses_to", "Stage 5"),
            ("Stage 1", "requires_action", "CVD Risk Reduction"),
            ("Stage 2", "requires_action", "Slow Progression"),
            ("Stage 3", "requires_action", "Manage Complications"),
            ("Stage 4", "requires_action", "Nephrology Referral"),
            ("Stage 4", "requires_action", "Dialysis Preparation"),
            ("Stage 5", "requires_action", "Kidney Replacement"),
            ("Hypertension", "accelerates", "Stage 3"),
            ("Diabetes Mellitus", "primary_cause", "Stage 1")
        ]
        for s, r, t in edges:
            self.add_edge(s, r, t)

    def ingest_vlm_triples(self, triples: List[Dict[str, str]]):
        """Ingest relationships extracted from VLM document parsing."""
        for t in triples:
            src = t.get("from")
            rel = t.get("relation")
            tgt = t.get("to")
            if src and rel and tgt:
                self.add_edge(src, rel, tgt, weight=1.0)
        self.save_to_outputs()

    def find_paths(self, start_entity: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Perform standard breadth-first search for multi-hop paths."""
        paths = []
        queue = [([start_entity], [])]

        while queue:
            node_path, edge_path = queue.pop(0)
            curr = node_path[-1]

            if len(node_path) > 1:
                paths.append({
                    "nodes": list(node_path),
                    "relations": list(edge_path),
                    "length": len(edge_path)
                })

            if len(edge_path) >= max_hops:
                continue

            for edge in self.adj.get(curr, []):
                nxt = edge["target"]
                if nxt not in node_path:
                    queue.append((node_path + [nxt], edge_path + [edge["relation"]]))

        return paths

    def query_subgraph(self, entities: List[str], max_hops: int = 2) -> Dict[str, Any]:
        """Extract a relevant neighborhood subgraph around selected entities."""
        sub_nodes = set()
        sub_edges = []
        lookup = {k.lower(): k for k in self.nodes.keys()}

        seed_keys = []
        for e in entities:
            e_clean = e.lower().strip()
            for k_lower, original_k in lookup.items():
                if e_clean in k_lower or k_lower in e_clean:
                    seed_keys.append(original_k)

        for seed in set(seed_keys):
            sub_nodes.add(seed)
            paths = self.find_paths(seed, max_hops=max_hops)
            for p in paths:
                nodes = p["nodes"]
                rels = p["relations"]
                for i in range(len(rels)):
                    u = nodes[i]
                    v = nodes[i+1]
                    rel = rels[i]
                    sub_nodes.add(u)
                    sub_nodes.add(v)
                    edge_dict = {"source": u, "target": v, "relation": rel}
                    if edge_dict not in sub_edges:
                        sub_edges.append(edge_dict)

        nodes_data = [self.nodes[nid] for nid in sub_nodes if nid in self.nodes]
        return {
            "nodes": nodes_data,
            "links": sub_edges
        }

    def to_web_dict(self) -> Dict[str, Any]:
        """Serialize complete graph for frontend visualization."""
        links = []
        for src, edges in self.adj.items():
            for e in edges:
                links.append({
                    "source": src,
                    "target": e["target"],
                    "relation": e["relation"]
                })
        return {
            "nodes": list(self.nodes.values()),
            "links": links
        }

    def save_to_outputs(self):
        """Save graph structure to outputs directory."""
        try:
            out_file = OUTPUTS_DIR / "knowledge_graph.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(self.to_web_dict(), f, indent=2)
        except Exception:
            pass

kg_instance = ClinicalKnowledgeGraph()
kg_instance.save_to_outputs()
