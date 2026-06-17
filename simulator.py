"""
NeuroCache — Cache Simulation Engine
Uses numpy for matrix operations and real LRU simulation.
"""
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal
import time


Pattern = Literal["row", "col", "tile", "random", "zigzag"]


@dataclass
class SimResult:
    hits: int
    misses: int
    hit_rate: float
    avg_latency_ns: float
    bandwidth_gbs: float
    eff_gflops: float
    timeline: list[int]          # 1=hit, 0=miss, sampled to ≤200 pts
    eviction_counts: list[int]   # evictions over time buckets
    working_set_kb: float
    compulsory_misses: int
    capacity_misses: int
    conflict_misses: int
    hotspot_map: list[float]     # 16×16 flattened heat map of access frequency


class LRUCache:
    """Simple set-associative LRU cache simulator."""
    def __init__(self, num_lines: int):
        self.num_lines = num_lines
        self.cache: OrderedDict[int, int] = OrderedDict()  # lineAddr -> load_time
        self.time = 0
        self.evictions = 0

    def access(self, line_addr: int) -> bool:
        self.time += 1
        if line_addr in self.cache:
            self.cache.move_to_end(line_addr)
            return True
        else:
            if len(self.cache) >= self.num_lines:
                self.cache.popitem(last=False)
                self.evictions += 1
            self.cache[line_addr] = self.time
            return False


def generate_accesses(n: int, pattern: Pattern, block_size: int = 0) -> np.ndarray:
    """Generate flat element indices for an N×N matrix access pattern."""
    if pattern == "row":
        return np.arange(n * n, dtype=np.int32)

    elif pattern == "col":
        indices = np.arange(n * n, dtype=np.int32).reshape(n, n)
        return indices.T.flatten()

    elif pattern == "tile":
        T = block_size if block_size > 0 else max(8, int(np.sqrt(n)))
        out = []
        for br in range(0, n, T):
            for bc in range(0, n, T):
                for r in range(br, min(br + T, n)):
                    for c in range(bc, min(bc + T, n)):
                        out.append(r * n + c)
        return np.array(out, dtype=np.int32)

    elif pattern == "zigzag":
        out = []
        for r in range(n):
            row = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
            for c in row:
                out.append(r * n + c)
        return np.array(out, dtype=np.int32)

    elif pattern == "random":
        rng = np.random.default_rng(42)
        return rng.integers(0, n * n, size=n * n, dtype=np.int32)

    return np.arange(n * n, dtype=np.int32)


def classify_misses(accesses: np.ndarray, words_per_line: int, num_lines: int):
    """Classify misses into compulsory / capacity / conflict (simplified)."""
    line_addrs = accesses // words_per_line
    unique_lines = np.unique(line_addrs)
    compulsory = len(unique_lines)          # first-ever touch
    capacity = max(0, len(line_addrs) - compulsory - max(0, len(unique_lines) - num_lines) * 2)
    conflict = max(0, len(line_addrs) - compulsory - capacity)
    return compulsory, capacity, conflict


def run_simulation(
    mat_size: int = 64,
    cache_kb: int = 32,
    cache_line_bytes: int = 64,
    pattern: Pattern = "row",
) -> SimResult:
    N = mat_size
    words_per_line = cache_line_bytes // 4   # float32
    num_lines = (cache_kb * 1024) // cache_line_bytes
    working_set_kb = (N * N * 4) / 1024

    accesses = generate_accesses(N, pattern)
    cache = LRUCache(num_lines)

    hits = 0
    misses = 0
    eviction_buckets = []
    BUCKETS = 20
    bucket_size = max(1, len(accesses) // BUCKETS)
    prev_evictions = 0

    SAMPLE = max(1, len(accesses) // 200)
    timeline = []

    for i, addr in enumerate(accesses):
        line_addr = int(addr) // words_per_line
        is_hit = cache.access(line_addr)
        if is_hit:
            hits += 1
        else:
            misses += 1

        if i % SAMPLE == 0:
            timeline.append(1 if is_hit else 0)

        if (i + 1) % bucket_size == 0:
            eviction_buckets.append(cache.evictions - prev_evictions)
            prev_evictions = cache.evictions

    total = hits + misses
    hit_rate = hits / total if total else 0
    avg_latency = hit_rate * 4.0 + (1 - hit_rate) * 100.0
    total_time_ns = total * avg_latency
    bandwidth = (total * 4) / total_time_ns if total_time_ns > 0 else 0
    eff_gflops = (2 * N ** 3) / total_time_ns if total_time_ns > 0 else 0

    compulsory, capacity, conflict = classify_misses(accesses, words_per_line, num_lines)

    # Build 16×16 hotspot heatmap
    hm_size = 16
    hm = np.zeros(hm_size * hm_size, dtype=np.float64)
    bins = np.floor(accesses * (hm_size * hm_size) / (N * N)).astype(np.int32)
    bins = np.clip(bins, 0, hm_size * hm_size - 1)
    np.add.at(hm, bins, 1)
    hm_max = hm.max() or 1
    hotspot_map = (hm / hm_max).tolist()

    return SimResult(
        hits=int(hits),
        misses=int(misses),
        hit_rate=float(hit_rate),
        avg_latency_ns=float(avg_latency),
        bandwidth_gbs=float(bandwidth),
        eff_gflops=float(eff_gflops),
        timeline=timeline,
        eviction_counts=eviction_buckets,
        working_set_kb=float(working_set_kb),
        compulsory_misses=int(compulsory),
        capacity_misses=int(capacity),
        conflict_misses=int(conflict),
        hotspot_map=hotspot_map,
    )


def sweep_matrix_sizes(
    cache_kb: int, cache_line_bytes: int, patterns: list[Pattern]
) -> dict:
    """Run simulation across matrix sizes for comparison chart."""
    sizes = [16, 32, 64, 128, 256]
    out: dict[str, list] = {p: [] for p in patterns}
    bw: dict[str, list] = {p: [] for p in patterns}
    for p in patterns:
        for s in sizes:
            r = run_simulation(s, cache_kb, cache_line_bytes, p)
            out[p].append(round(r.hit_rate, 4))
            bw[p].append(round(r.bandwidth_gbs, 4))
    return {"hit_rates": out, "bandwidths": bw, "sizes": sizes}
