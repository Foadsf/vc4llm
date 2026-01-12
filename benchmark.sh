#!/bin/bash
# benchmark.sh - Run on actual Raspberry Pi

MODEL="SmolLM2-135M-Instruct-Q8_0.gguf"
PROMPT="Once upon a time"
TOKENS=50

echo "=== VC4LLM Benchmark ==="
echo "Model: $MODEL"
echo "Tokens: $TOKENS"
echo ""

# CPU benchmark
echo "--- CPU Mode (4 threads) ---"
./vc4llm -m $MODEL -p "$PROMPT" -n $TOKENS -t 4

echo ""

# GPU benchmark
echo "--- GPU Mode ---"
sudo ./vc4llm -m $MODEL -p "$PROMPT" -n $TOKENS -t 4 --gpu

echo ""
echo "=== Benchmark Complete ==="
