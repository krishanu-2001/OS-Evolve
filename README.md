<h1 align="center">
  <img width="180" alt="image" src="https://github.com/user-attachments/assets/f282c950-039a-41ce-9894-cc1c0fd2f461" /><br/>
  <b><code>OSEvolve</code>: Workload-Aware Evolution of CPU Scheduling Algorithms</b><br>
</h1>

## Overview

**OS-Evolve** is an automated framework that leverages Large Language Models (LLMs) and evolutionary algorithms to generate, mutate, and optimize C++ CPU scheduling policies. By integrating with Google's [ghOSt](https://github.com/google/ghost-userspace) userspace scheduling framework, OS-Evolve bypasses the operational hazards of kernel development, enabling safe experimentation and rapid iteration of scheduler implementations.

Main Fork: https://github.com/yogasrivarshan/OS-Evolve

### The Problem: One-Size-Fits-All Scheduling

The Linux Completely Fair Scheduler (CFS) is designed for "best average case" performance across generic workloads. However, modern datacenters often feature specialized, predictable workloads—either short interactive bursts or long computation cycles—where tailored schedulers like [Shinjuku](https://www.usenix.org/conference/nsdi19/presentation/kaffes-shinjuku) and [Concord](https://dl.acm.org/doi/10.1145/3600006.3613163) have demonstrated massive gains in tail latency and throughput.

**The Challenge:** Manually designing these specialized schedulers requires deep domain expertise, extensive trace analysis, and operational complexity. Implementing and updating policies in the kernel is expensive and risky.

### Our Solution: Evolutionary Policy Optimization

OS-Evolve treats CPU schedulers as **white-box C++ programs** that can be iteratively improved through:

- **LLM-Driven Mutations**: Large language models propose code modifications to scheduler implementations, guided by performance feedback
- **Evolutionary Search**: A population of scheduler candidates evolves over generations, with the best performers serving as parents for the next generation
- **ghOSt Integration**: Policies are compiled and evaluated in userspace, avoiding kernel modifications and enabling hot-swapping of schedulers
- **Global Knowledge Base (GKB)**: A permanent archive of "veteran" policies that have proven themselves on specific workloads, enabling knowledge transfer and monotonic improvement

### Key Results

Our experiments demonstrate that OS-Evolve successfully discovers scheduling policies that outperform baseline implementations:

| Workload | Num Threads | Num CPUs | Initial Score (TPS) | Best Evolved Score | Improvement |
|----------|-------------|----------|---------------------|--------------------|--------------------|
| SysBench | 4 | 5 | 1234.56 | 1240.98 | +0.5% |
| SysBench | 32 | 5 | 1504.80 | 1517.00 | +0.8% |
| SysBench | 32 | 10 | 4621.63 | 4737.81 | **+2.5%** |
| SysBench | 64 | 5 | 1502.34 | 1541.11 | **+2.6%** |

**Head-of-Line Blocking Tests:** The evolutionary process successfully pushes the Pareto frontier of latency vs. throughput, with specific generations (Gen 5 and Gen 8) significantly improving throughput by prioritizing fast threads and mitigating queue blocking.

**Evolved Heuristics:** Analysis reveals a hierarchical evolution pattern:
- **Early Generations**: Focus on tuning hyperparameters (e.g., load imbalance thresholds)
- **Advanced Generations**: Implement sophisticated algorithmic improvements:
  - Lockless runqueue load checks to avoid cross-CPU lock contention
  - Tiered CPU placement logic to prevent cache thrashing
  - Smart handling of idle vs. busy CPU states

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OS-Evolve Framework                       │
│                                                              │
│  ┌──────────────┐      ┌─────────────┐      ┌────────────┐ │
│  │ Evolutionary │─────▶│ LLM Mutator │─────▶│  Compiler  │ │
│  │    Engine    │      │  (GPT-4)    │      │  (Bazel)   │ │
│  └──────────────┘      └─────────────┘      └────────────┘ │
│         │                                            │       │
│         │                                            ▼       │
│         │                                    ┌────────────┐ │
│         │                                    │   ghOSt    │ │
│         │                                    │  Userspace │ │
│         │                                    │  Scheduler │ │
│         │                                    └────────────┘ │
│         │                                            │       │
│         │                                            ▼       │
│         │                                    ┌────────────┐ │
│         │                                    │ Benchmarks │ │
│         │                                    │ (SysBench, │ │
│         │                                    │    HoL)    │ │
│         │                                    └────────────┘ │
│         │                                            │       │
│         │              Performance Metrics           │       │
│         └────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Global Knowledge Base (SQLite)              │   │
│  │  Archives veteran policies for knowledge transfer    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Future Vision: Self-Adapting OS

Our roadmap includes closing the loop between offline evolution and online execution:

1. **Online Hot-Swap**: When a new workload arrives, query the GKB for the optimal policy and hot-swap it into the running scheduler without kernel modification
2. **Offline Feedback**: Live metrics (SLO violations, latency spikes) trigger offline evolution sessions to discover improved policies
3. **Continuous Improvement**: Results are fed back into the GKB, creating a self-learning OS that monotonically improves over time

## Installation & Quick Start 🚀

```bash
# Clone the repository
git clone https://github.com/SakanaAI/ShinkaEvolve
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install Shinka
cd ShinkaEvolve
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Run your first evolution experiment
See makefile for instructions
```
