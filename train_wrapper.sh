#!/bin/bash
set -e

echo "========================================="
echo "AKASH TRAINING JOB STARTED"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU detected, using CPU)"
echo "========================================="

python3 train.py

EXIT_CODE=$?

echo "========================================="
echo "TRAINING COMPLETE — exit code: $EXIT_CODE"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Output files:"
ls -lh /output/
echo "========================================="

echo "$EXIT_CODE" > /output/TRAINING_DONE

echo "Container staying alive for file retrieval (15 min)..."
sleep 900

exit $EXIT_CODE
