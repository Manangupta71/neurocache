## **Neural Cache Sim**

A high-performance visualizer designed to demystify how matrix multiplication—the backbone of neural networks—interacts with CPU cache hierarchies. This tool bridges the gap between high-level linear algebra and low-level systems engineering.

### **Core Capabilities**
* **Hardware Sandbox:** Manually tune matrix dimensions, total cache size, and cache line bytes to match specific hardware profiles.
* **Real-Time Animation:** Watch a live heat map of hits and misses across **128 cache lines** as the algorithm traverses memory.
* **Access Pattern Benchmarking:** Compare the efficiency of **Row-Major**, **Column-Major**, and **Tiled (Blocked)** patterns.
* **Performance Analytics:** * Track hit rate vs. matrix size.
    * Measure memory bandwidth consumption.
    * Calculate estimated **GFLOPS/s** efficiency.

---

### **The Performance Gap**
Modern deep learning performance is often limited by memory latency rather than raw compute power. This simulation illustrates how "cache-friendly" code (like tiling) minimizes data movement and maximizes throughput.

$$\text{Efficiency} \propto \frac{\text{Arithmetic Intensity}}{\text{Memory Latency}}$$



### **Usage**
Use this tool to visualize why naive matrix multiplication fails on large datasets and how blocking strategies keep data in L1/L2 cache to prevent "cache thrashing." Ideal for students of computer architecture and developers optimizing inference engines.
