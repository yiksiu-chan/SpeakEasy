#!/bin/bash
set -euo pipefail

FRAMEWORKS=("baseline_dr" "speakeasy_dr" "baseline_gcg" "speakeasy_gcg" "baseline_tap" "speakeasy_tap")
MODEL="vllm:meta-llama/Llama-3.3-70B-Instruct"  # or "openai:gpt-4o"

BENCHMARKS=(
    "data/advbench/data.json"
    "data/harmbench/data.json"
    "data/sorry-bench/data.json"
    "data/med-safety-bench/data.json"
)

for benchmark in "${BENCHMARKS[@]}"; do
    for framework in "${FRAMEWORKS[@]}"; do
        echo "Running $framework on $benchmark"
        speakeasy run \
            --data-dir "$benchmark" \
            --save-dir results/ \
            --model "$MODEL" \
            --frameworks "$framework"
    done
done
