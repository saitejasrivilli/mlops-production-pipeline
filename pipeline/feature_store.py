"""
Lightweight Redis-backed feature store for the ML pipeline.
Stores: user features, item features, precomputed embeddings.
Falls back to in-memory dict if Redis unavailable.
"""
import json
import hashlib
import time
from typing import Any, Optional


class FeatureStore:
    """Key-value store for ML features backed by Redis with in-memory fallback."""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self._ttl = ttl
        self._backend = "redis"
        self._hits = 0
        self._misses = 0
        self._total_latency_ms = 0.0
        self._n_calls = 0

        try:
            import redis as _redis
            self._client = _redis.from_url(redis_url, socket_connect_timeout=1)
            self._client.ping()
        except Exception:
            self._client = None
            self._backend = "memory"
            self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _now(self) -> float:
        return time.monotonic()

    def _record(self, latency_ms: float) -> None:
        self._total_latency_ms += latency_ms
        self._n_calls += 1

    # ── Core API ────────────────────────────────────────────────────────────

    def put(self, key: str, features: dict, ttl: int = None) -> bool:
        """Store a feature dict under `key`. Returns True on success."""
        effective_ttl = ttl if ttl is not None else self._ttl
        t0 = time.perf_counter()
        try:
            if self._client is not None:
                self._client.setex(key, effective_ttl, json.dumps(features))
            else:
                expires_at = self._now() + effective_ttl
                self._store[key] = (features, expires_at)
            self._record((time.perf_counter() - t0) * 1000)
            return True
        except Exception:
            return False

    def get(self, key: str) -> Optional[dict]:
        """Retrieve features by key. Returns None on cache miss or expiry."""
        t0 = time.perf_counter()
        result = None
        try:
            if self._client is not None:
                raw = self._client.get(key)
                result = json.loads(raw) if raw is not None else None
            else:
                entry = self._store.get(key)
                if entry is not None:
                    value, expires_at = entry
                    if self._now() < expires_at:
                        result = value
                    else:
                        del self._store[key]
        except Exception:
            result = None

        self._record((time.perf_counter() - t0) * 1000)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put_batch(self, records: list[dict], key_field: str = "id") -> int:
        """Bulk-insert records. Returns count of successfully stored items."""
        stored = 0
        if self._client is not None:
            pipe = self._client.pipeline(transaction=False)
            for rec in records:
                key = str(rec.get(key_field, ""))
                if key:
                    pipe.setex(key, self._ttl, json.dumps(rec))
                    stored += 1
            pipe.execute()
        else:
            for rec in records:
                key = str(rec.get(key_field, ""))
                if key and self.put(key, rec):
                    stored += 1
        return stored

    def get_batch(self, keys: list[str]) -> list[Optional[dict]]:
        """Bulk-fetch by keys. Returns list aligned with input keys."""
        if self._client is not None:
            pipe = self._client.pipeline(transaction=False)
            for k in keys:
                pipe.get(k)
            raws = pipe.execute()
            results = []
            for raw in raws:
                if raw is not None:
                    results.append(json.loads(raw))
                    self._hits += 1
                else:
                    results.append(None)
                    self._misses += 1
            return results
        else:
            return [self.get(k) for k in keys]

    def delete(self, key: str) -> bool:
        """Remove a key. Returns True if the key existed."""
        try:
            if self._client is not None:
                return bool(self._client.delete(key))
            else:
                existed = key in self._store
                self._store.pop(key, None)
                return existed
        except Exception:
            return False

    def stats(self) -> dict:
        """Returns: {backend, n_keys, hit_rate, avg_latency_ms}"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        avg_latency = self._total_latency_ms / self._n_calls if self._n_calls > 0 else 0.0

        if self._client is not None:
            try:
                n_keys = self._client.dbsize()
            except Exception:
                n_keys = -1
        else:
            now = self._now()
            n_keys = sum(1 for _, (_, exp) in self._store.items() if now < exp)

        return {
            "backend": self._backend,
            "n_keys": n_keys,
            "hit_rate": round(hit_rate, 4),
            "avg_latency_ms": round(avg_latency, 3),
            "total_gets": total,
            "hits": self._hits,
            "misses": self._misses,
        }


class EmbeddingStore(FeatureStore):
    """Stores precomputed text embeddings with numpy-compatible serialization."""

    def put_embedding(self, text: str, embedding: list[float]) -> str:
        """Store embedding keyed by sha256(text)[:16]. Returns the key."""
        key = self._text_key(text)
        self.put(key, {"text_hash": key, "embedding": embedding})
        return key

    def get_embedding(self, text: str) -> Optional[list[float]]:
        """Retrieve embedding for text, or None on miss."""
        key = self._text_key(text)
        record = self.get(key)
        if record is not None:
            return record.get("embedding")
        return None

    def get_or_compute(self, text: str, compute_fn) -> list[float]:
        """Cache-aside pattern: return cached embedding or compute, cache, and return."""
        cached = self.get_embedding(text)
        if cached is not None:
            return cached
        embedding = compute_fn(text)
        # Normalise to plain list for JSON serialisation
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        self.put_embedding(text, embedding)
        return embedding

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
