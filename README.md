# NeuroCache

A cache simulation engine that models how matrix traversal patterns interact with CPU cache hierarchies — built to illustrate why "cache-friendly" code (like tiling) matters for ML and numerical workloads.

**[Live demo →](https://manangupta71.github.io/neurocache/)**

## What it does

NeuroCache simulates a set-associative LRU cache against five memory access patterns (row-major, column-major, tiled, zigzag, random) and visualizes the result: hit/miss timeline, live cache state, miss classification (compulsory / capacity / conflict), access hotspot heatmap, and bandwidth/GFLOPS estimates across matrix sizes.

The core idea: efficiency is roughly proportional to arithmetic intensity divided by memory latency. Code that accesses memory sequentially (row-major) keeps hit rates high; code that strides across memory (column-major on a large matrix) thrashes the cache and tanks throughput, even though both perform the same number of FLOPs.

## Two implementations

This repo contains two versions of the same simulation logic:

**`docs/index.html`** — a self-contained JS port, used for the GitHub Pages live demo. No backend, runs entirely in-browser.

**`simulator.py` + `app.py`** — the reference implementation. Uses `numpy` for vectorized access-pattern generation and an `OrderedDict`-based LRU eviction model, served via Flask. This is the "real" engine; the JS version mirrors its logic so the demo can run on static hosting.

## Running the Python version locally

```bash
git clone https://github.com/Manangupta71/neurocache.git
cd neurocache
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860`.

## Core capabilities

- Tune matrix size, total cache size, and cache line size to model different hardware profiles
- Live animated visualization of hits/misses across 128 cache lines as the simulation runs
- Compare hit rate and bandwidth across row-major, column-major, tiled, zigzag, and random access patterns
- Miss classification: compulsory (cold) vs capacity vs conflict misses
- Estimated latency, memory bandwidth, and effective GFLOPS/s derived from the simulated hit rate

## Stack

Python · NumPy · Flask · Chart.js
