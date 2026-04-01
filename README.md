# gicolm

A pure Go implementation of [picolm](https://github.com/RightNow-AI/picolm) — ultra-lightweight LLM inference engine.

## Features

- **GGUF model loader** with mmap support
- **Quantized inference**: Q2_K, Q3_K, Q4_K, Q6_K, Q8_0, Q4_0, F16, F32
- **SIMD-accelerated** vector operations (auto-vectorized via loop unrolling)
- **Transformer forward pass** with flash attention (online softmax) and FP16 KV cache
- **BPE tokenizer** with SentencePiece support
- **Sampling**: temperature scaling, top-p (nucleus) sampling
- **JSON grammar-constrained** sampling for structured output
- **KV cache persistence** for skipping prompt prefill

## Build

```bash
go build -o gicolm .
```

## Usage

```bash
./gicolm <model.gguf> [options]
```

### Generation options

| Flag | Description | Default |
|------|-------------|---------|
| `-p <prompt>` | Input prompt (or pipe via stdin) | |
| `-n <int>` | Max tokens to generate | 256 |
| `-t <float>` | Temperature (0 = greedy) | 0.8 |
| `-k <float>` | Top-p / nucleus sampling | 0.9 |
| `-s <int>` | RNG seed | 42 |
| `-c <int>` | Context length override | |
| `-j <int>` | Number of threads | num CPUs |

### Advanced options

| Flag | Description |
|------|-------------|
| `--json` | Grammar-constrained JSON output mode |
| `--cache <file>` | KV cache file (saves/loads prompt state) |

### Examples

```bash
# Basic generation
./gicolm model.gguf -p "Hello, world" -n 128

# Greedy decoding
./gicolm model.gguf -p "What is Go?" -t 0

# JSON output
./gicolm model.gguf -p '{"name":' --json -n 64

# Pipe via stdin
echo "Once upon a time" | ./gicolm model.gguf -n 256
```

## Supported models

Any GGUF-format model compatible with the Llama architecture. Tested with:

- TinyLlama 1.1B Chat (Q4_K_M)

## Performance

Core vector operations benchmarked on Intel i5-11500 (Rocket Lake):

| Operation | Throughput |
|-----------|------------|
| VecDotF32 (dot product) | ~4200 MB/s |
| ElemwiseMul (element-wise multiply) | ~8700 MB/s |
| VecAdd (vector add) | ~9300 MB/s |
| RmsNorm (layer normalization) | ~4700 MB/s |

SIMD acceleration is achieved via 4x loop unrolling, enabling the Go compiler's auto-vectorization (SSE/AVX) on amd64. Platform-specific implementations are selected at build time using build tags.

## License

MIT
