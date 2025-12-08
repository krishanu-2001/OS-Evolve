# Sysbench Evolution Setup

This directory contains the evolution setup for optimizing the ghOSt CFS scheduler using sysbench CPU workloads.

## Prerequisites

1. **ghOSt kernel installed** (kernel 5.11+)
2. **ghost-userspace** cloned and set up
3. **Bedrock API key** configured in `.env` file
4. **sysbench** installed (`apt-get install sysbench` or equivalent)
5. **Python 3.10+** with required packages

## Setup Steps

### 1. Install ghOSt Userspace Assets

```bash
cd /mydata/OS-Evolve
make install-ghost-userspace
```

This will:
- Copy test files to `ghost-userspace/`
- Copy BUILD configuration
- Build `agent_cfs_test`, `cfs_hol_test`, and `cfs_sysbench_test`

### 2. Verify Test Binary

```bash
cd ghost-userspace
sudo bazel build -c opt cfs_sysbench_test
sudo ./bazel-bin/cfs_sysbench_test --create_enclave_and_agent --ghost_cpus=1-5
```

You should see:
- Enclave creation messages
- Sysbench process being moved to ghOSt
- Metrics written to `metrics/cfs_sysbench.csv`

### 3. Configure Environment

Ensure `.env` file in `/mydata/OS-Evolve/` contains:
```
BEDROCK_API_KEY=your_key_here
AWS_REGION_NAME=us-west-2
```

### 4. Verify Initial Code

Check that `initial.cc` exists and has EVOLVE-BLOCK markers:
```bash
grep -n "EVOLVE-BLOCK" initial.cc
```

You should see multiple `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers.

## Running Evolution

### Quick Test (1 generation)

```bash
cd /mydata/OS-Evolve/examples/scheduler_sysbench
python run_evo.py --num_generations=1 --results_dir=results_cfs_scheduler
```

### Full Evolution (60 generations)

```bash
python run_evo.py --num_generations=60 --results_dir=results_cfs_scheduler
```

## Results

Results are stored in `results_cfs_scheduler/`:
- `gen_N/main.cpp` - Evolved code for generation N
- `gen_N/results/metrics.json` - Performance metrics
- `gen_N/results/correct.json` - Build/test success status
- `best/main.cpp` - Best candidate found so far
- `evolution_db.sqlite` - Database of all candidates

## Fitness Metrics

The default fitness metric is `events_per_sec` (higher is better). You can change it in `evaluate.py`:

- `events_per_sec`: Sysbench events per second
- `avg_latency`: Inverse of average latency (higher is better)
- `throughput_latency_ratio`: Combined metric

## Troubleshooting

### Build Failures
- Check that `ghost-userspace` is properly set up
- Verify Bazel version matches `.bazelversion` (7.1.1)
- Ensure all dependencies are installed

### Test Failures
- Verify ghOSt kernel is loaded: `lsmod | grep ghost`
- Check enclave exists: `ls /sys/fs/ghost/enclave_*`
- Verify sysbench is installed: `which sysbench`

### Evolution Not Running
- Check `.env` file has correct API keys
- Verify Python version: `python --version` (should be 3.10+)
- Check database permissions: `ls -la evolution_db.sqlite`

## File Structure

```
scheduler_sysbench/
├── initial.cc              # Starting scheduler code with EVOLVE-BLOCK markers
├── run_evo.py              # Evolution configuration and entry point
├── evaluate.py             # Evaluation harness (build, test, parse metrics)
├── results_cfs_scheduler/  # Evolution results (created at runtime)
└── README.md               # This file
```

## Differences from HoL Evolution

| Aspect | HoL | Sysbench |
|--------|-----|----------|
| Test Binary | `cfs_hol_test` | `cfs_sysbench_test` |
| Workload | Synthetic ghost threads | Real sysbench process |
| Metrics | Queueing delay, throughput | Events/sec, latency |
| Fitness | `throughput` or `latency_p95` | `events_per_sec` (default) |

Everything else (evolution framework, LLM integration, database) is identical.

