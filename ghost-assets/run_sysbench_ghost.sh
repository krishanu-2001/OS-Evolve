#!/bin/bash
# Script to run sysbench under ghOSt scheduler
# Usage: ./run_sysbench_ghost.sh <num_threads> <ghost_cpus>
# Example: ./run_sysbench_ghost.sh 1 "1-5"

NUM_THREADS=${1:-1}
GHOST_CPUS=${2:-"1-5"}

# Check if sysbench is installed
if ! command -v sysbench &> /dev/null; then
    echo "Error: sysbench is not installed"
    exit 1
fi

# Run sysbench CPU test
# Note: This runs sysbench as a regular process, not as a ghOSt thread
# To actually use ghOSt scheduling, the process needs to be moved to the ghOSt enclave
# This is a simplified version - for full ghOSt integration, you'd need to:
# 1. Create a ghOSt thread that runs sysbench
# 2. Or use taskset to pin to ghost CPUs and rely on the scheduler

sysbench cpu \
    --threads=$NUM_THREADS \
    --cpu-max-prime=20000 \
    --time=10 \
    run

