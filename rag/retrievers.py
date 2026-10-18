"""
Standard Multi-Retriever and Reciprocal Rank Fusion (RRF) Module.
Implements:
1. Lexical Retriever (Okapi BM25)
2. Dense Semantic Retriever (text-embedding-3-large)
3. Relational Knowledge Graph Retriever
4. Reciprocal Rank Fusion (RRF) for blending results.
All comments and code are strictly in English.
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
from embeddings import embedder, cosine_similarity
from knowledge_graph import kg_instance

# Standard tokenization regex keeping alphanumeric words, numbers, and comparison signs
TOKEN_REGEX = re.compile(r"[A-Za-z][A-Za-z0-9+\-_/]*|\d+(?:\.\d+)?|[><=]+")

def tokenize_clinical_text(text: str) -> List[str]:
    """Tokenize text into lowercase tokens."""
    return [t.lower() for t in TOKEN_REGEX.findall(str(text or ""))]

class BM25Retriever:
    """Standard Okapi BM25 lexical retriever."""
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_lens = []
        self.postings = defaultdict(list)
        self.df = Counter()
        self.avg_dl = 0.0
        self._build_index()

    def _build_index(self):
        total_len = 0
        for idx, c in enumerate(self.chunks):
            tokens = tokenize_clinical_text(c["text"])
            dl = len(tokens)
            self.doc_lens.append(dl)
            total_len += dl

            tf = Counter(tokens)
            for token, freq in tf.items():
                self.postings[token].append((idx, freq))
                self.df[token] += 1

        n_docs = max(len(self.chunks), 1)
        self.avg_dl = total_len / n_docs

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        tokens = tokenize_clinical_text(query)
        if not tokens:
            return []

        scores = defaultdict(float)
        n_docs = len(self.chunks)

        for token in tokens:
            if token not in self.postings:
                continue
            df = self.df[token]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_idx, tf in self.postings[token]:
                dl = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avg_dl or 1.0)))
                scores[doc_idx] += idf * (numerator / denominator)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_idx, score in ranked:
            if score > 0:
                chunk = dict(self.chunks[doc_idx])
                chunk["score"] = float(score)
                chunk["retriever"] = "bm25"
                results.append(chunk)
        return results

class DenseVectorRetriever:
    """Standard semantic vector retriever using cosine similarity."""
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.embeddings = []
        self._build_index()

    def _build_index(self):
        texts = [c["text"] for c in self.chunks]
        if texts:
            self.embeddings = embedder.embed_texts(texts)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
        q_emb = embedder.embed_query(query)
        scored = []
        for idx, (chunk, c_emb) in enumerate(zip(self.chunks, self.embeddings)):
            sim = cosine_similarity(q_emb, c_emb)
            scored.append((idx, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, sim in scored[:top_k]:
            chunk = dict(self.chunks[doc_idx])
            chunk["score"] = float(sim)
            chunk["retriever"] = "dense_vector"
            results.append(chunk)
        return results

class KGRetriever:
    """Standard Knowledge Graph entity & relationship retriever."""
    def __init__(self, chunks: List[Dict[str, Any]], kg=kg_instance):
        self.chunks = chunks
        self.kg = kg

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        tokens = set(tokenize_clinical_text(query))
        matched_entities = []
        for nid in self.kg.nodes.keys():
            n_tokens = set(tokenize_clinical_text(nid))
            if n_tokens and n_tokens.issubset(tokens):
                matched_entities.append(nid)

        subgraph = self.kg.query_subgraph(matched_entities, max_hops=2)
        sub_text_tokens = set()
        for node in subgraph["nodes"]:
            sub_text_tokens.update(tokenize_clinical_text(node["label"]))
        for link in subgraph["links"]:
            sub_text_tokens.update(tokenize_clinical_text(link["relation"]))

        scored = []
        for idx, c in enumerate(self.chunks):
            c_tokens = set(tokenize_clinical_text(c["text"]))
            overlap = len(c_tokens.intersection(sub_text_tokens))
            if overlap > 0:
                scored.append((idx, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, overlap_score in scored[:top_k]:
            chunk = dict(self.chunks[doc_idx])
            chunk["score"] = float(overlap_score)
            chunk["retriever"] = "knowledge_graph"
            chunk["kg_subgraph"] = subgraph
            results.append(chunk)
        return results

class FusionRetriever:
    """Standard Reciprocal Rank Fusion (RRF) rank aggregation."""
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseVectorRetriever(chunks)
        self.kg = KGRetriever(chunks)

    def retrieve(self, query: str, top_k: int = 4, weights: Dict[str, float] = None, k_rrf: int = 60) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        weights = weights or {"dense": 1.0, "bm25": 0.8, "kg": 0.9}

        dense_hits = self.dense.retrieve(query, top_k=top_k*2)
        bm25_hits = self.bm25.retrieve(query, top_k=top_k*2)
        kg_hits = self.kg.retrieve(query, top_k=top_k*2)

        rrf_scores = defaultdict(float)
        provenance = defaultdict(list)
        chunk_map = {}

        retriever_runs = [
            ("dense", dense_hits, weights.get("dense", 1.0)),
            ("bm25", bm25_hits, weights.get("bm25", 0.8)),
            ("kg", kg_hits, weights.get("kg", 0.9)),
        ]

        for name, hit_list, w in retriever_runs:
            for rank, hit in enumerate(hit_list, 1):
                cid = hit["chunk_id"]
                chunk_map[cid] = hit
                rrf_scores[cid] += w / (k_rrf + rank)
                provenance[cid].append({
                    "retriever": name,
                    "rank": rank,
                    "raw_score": round(hit.get("score", 0.0), 4)
                })

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_docs = []
        for cid in sorted_cids[:top_k]:
            doc = dict(chunk_map[cid])
            doc["rrf_score"] = round(rrf_scores[cid], 5)
            doc["provenance"] = provenance[cid]
            final_docs.append(doc)

        trace = {
            "query": query,
            "weights": weights,
            "dense_count": len(dense_hits),
            "bm25_count": len(bm25_hits),
            "kg_count": len(kg_hits),
            "total_fused": len(final_docs)
        }

        return final_docs, trace
