# MNIST Akash Training Test

A minimal PyTorch MNIST training job for testing the akash-train OpenClaw skill.

- Trains a simple CNN for 2 epochs
- Saves model weights to `/output/mnist_model.pt`
- Saves results JSON to `/output/results.json`
- Wrapper script signals completion and keeps container alive for file retrieval

## Quick test

```bash
# Build
docker build -t yourusername/mnist-akash-test:latest .

# Run locally (CPU)
docker run --rm -v $(pwd)/output:/output yourusername/mnist-akash-test:latest

# Push
docker push yourusername/mnist-akash-test:latest
```

## With OpenClaw

Message your bot:

> Train the MNIST model from github.com/yourusername/mnist-akash-test using an RTX 4090 on Akash. Docker image is yourusername/mnist-akash-test:latest
