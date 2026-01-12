# VC4LLM: VideoCore IV LLM Inference Engine

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%203B-red.svg)](https://www.raspberrypi.org/)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-orange.svg)](https://isocpp.org/)

**VC4LLM** is a lightweight, bare-metal-style Large Language Model (LLM) inference engine built from scratch for the **Raspberry Pi 3 Model B**. It is designed to squeeze maximum performance out of the Cortex-A53 CPU using NEON SIMD and explore the capabilities of the VideoCore IV GPU via VC4CL OpenCL.

## Overview

Running modern LLMs on older edge hardware like the Raspberry Pi 3B (1GB RAM, quad-core A53) is a significant challenge. Existing frameworks often have high overhead or lack specific optimizations for this architecture.

**VC4LLM** solves this by:

1. **Zero Dependencies**: No external BLAS, Torch, or Python runtime required. Just C++17.
2. **Hardware Optimization**: Custom NEON assembly-level intrinsics for matrix multiplication.
3. **GPU Support**: Direct targeting of the VideoCore IV QPU for compute, optimized for Q8_0 matrix multiplication.

This project serves as both a practical inference tool for small models (like SmolLM-135M) and a research platform for low-level optimization on the Raspberry Pi.

## Features

- **GGUF v3 Support**: Natively loads modern GGUF model files
- **NEON SIMD Acceleration**: Hand-tuned kernels for Q8_0 dot products and vector operations
- **Multi-threading**: Custom lightweight thread pool for parallelizing matrix operations
- **Memory Efficiency**: Uses `mmap` for instant model loading and OS-managed paging
- **BPE Tokenizer**: Full byte-level Byte Pair Encoding (GPT-2 style) support
- **Tied Embeddings**: Support for small models that share input/output embedding weights
- **Hybrid Compute**: Support for offloading layers to the VideoCore IV GPU via OpenCL

## Performance Benchmarks

**Hardware**: Raspberry Pi 3 Model B v1.2 (Quad-core Cortex-A53 @ 1.2GHz, 1GB RAM)  
**Model**: SmolLM2-135M-Instruct-Q8_0.gguf (138 MB)

| Configuration | Speed (tok/s) | Notes |
|:--------------|:-------------:|:------|
| Phase 1 (Scalar CPU) | 0.54 | Baseline naive C++ implementation |
| Phase 2 (NEON + 4 Threads) | 5.46 | 10x Speedup - SIMD + Parallelism |
| **Phase 3 (GPU OpenCL)** | **> 6.00** | **Optimal** - Faster than CPU |

## Requirements

### Hardware

- **Raspberry Pi 3 Model B / B+** (or Pi 2 v1.2 with Cortex-A53)
- **MicroSD Card**: Class 10 or UHS-I recommended for fast paging
- **Power Supply**: Reliable 5V 2.5A supply (critical for max CPU/GPU load)

### Software

- **OS**: Raspberry Pi OS (Legacy) **32-bit** (Bookworm or Bullseye)
- **Compiler**: GCC 8+ (needs C++17 support)
- **(Required for GPU)** VC4CL OpenCL implementation for VideoCore IV GPU acceleration

## Building

### 1. Clone the Repository

```bash
git clone https://github.com/Foadsf/vc4llm.git
cd vc4llm
```

### 2. Install Dependencies

```bash
sudo apt update
sudo apt install build-essential git

# Required for GPU support
sudo apt install ocl-icd-opencl-dev
```

### 3. Compile (GPU Optimized)

```bash
g++ -O3 -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard \
    -o vc4llm vc4llm.cpp -lpthread -lOpenCL
```

## Usage

### Basic Inference

```bash
./vc4llm -m SmolLM2-135M-Instruct-Q8_0.gguf -p "Once upon a time" -n 50 --gpu
```

### Command Line Options

```
./vc4llm [options]

Options:
  -m <file>    Path to GGUF model file (Required)
  -p <text>    Input prompt (Default: "Hello world")
  -n <int>     Number of tokens to generate (Default: 20)
  -t <int>     Number of CPU threads (Default: 4)
  -v           Verbose output (print model/layer info)
  --gpu        Enable OpenCL GPU acceleration (requires sudo)
```

### Example Output

```
$ sudo ./vc4llm -m SmolLM2-135M-Instruct-Q8_0.gguf -p "Once upon a time" -n 20 --gpu

Model mmapped: SmolLM2-135M-Instruct-Q8_0.gguf (138 MB)
Prompt: 'Once upon a time' (5 tokens)
Generating 20 tokens with 4 threads...
Inference init: dim=576, hidden=1536, head_dim=64, threads=4, gpu=ON
INT8 dot product: YES
Uploading weights to GPU...
Uploaded 219 tensors to GPU
You are a skilled and cunning pirate who has been sailing the seven seas for many years. You
Done. Time: 3.12s (6.41 tok/s)

=== Performance Breakdown ===
GPU kernel time: 2450.5 ms
GPU transfer time: 50.2 ms
CPU compute time: 619.3 ms
GPU calls: 120, CPU calls: 480
```

### GPU Mode

```bash
# GPU mode requires root access for /dev/mem (used by VC4CL)
sudo ./vc4llm -m SmolLM2-135M-Instruct-Q8_0.gguf -p "Hello" -n 20 --gpu
```

## Model Compatibility

VC4LLM supports **GGUF v3** models. Due to the 1GB RAM limit, you are restricted to small models (typically <500M parameters).

### Tested Models

| Model | Size | Status |
|:------|:----:|:------:|
| SmolLM2-135M-Instruct (Q8_0) | 138 MB | ✅ Recommended |
| SmolLM2-135M-Instruct (Q4_K_M) | ~80 MB | ⚠️ Q4 not yet supported |
| TinyLlama-1.1B (Q4_K_M) | ~600 MB | ⚠️ Requires swap, very slow |

### Downloading Models

```bash
# SmolLM2-135M (Recommended for Pi 3B)
wget https://huggingface.co/lmstudio-community/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf
```

Browse more models at [HuggingFace](https://huggingface.co/models?search=gguf).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        VC4LLM                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ GGUF Parser │→ │  Tokenizer  │→ │  Inference Engine   │  │
│  │   (mmap)    │  │    (BPE)    │  │                     │  │
│  └─────────────┘  └─────────────┘  │  ┌───────────────┐  │  │
│                                    │  │   Embedding   │  │  │
│  ┌─────────────────────────────┐   │  ├───────────────┤  │  │
│  │      Compute Backend        │   │  │  Transformer  │  │  │
│  │  ┌───────┐    ┌──────────┐  │   │  │    Layers     │  │  │
│  │  │ NEON  │    │  VC4CL   │  │   │  │  (RMSNorm,    │  │  │
│  │  │ SIMD  │    │  OpenCL  │  │   │  │   Attention,  │  │  │
│  │  │ (CPU) │    │  (GPU)   │  │   │  │   FFN, RoPE)  │  │  │
│  │  └───────┘    └──────────┘  │   │  ├───────────────┤  │  │
│  └─────────────────────────────┘   │  │    Output     │  │  │
│                                    │  │   (Argmax)    │  │  │
│                                    │  └───────────────┘  │  │
│                                    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Known Limitations

1. **Quantization Support**: Currently only supports **F32** and **Q8_0** weights. Q4_K support is planned.
2. **Context Length**: Fixed at model default (usually 2048), practical limit depends on available RAM.
3. **Platform**: Heavily optimized for 32-bit ARMv8. Requires modifications for x86 or 64-bit ARM.
4. **Sampling**: Currently uses greedy argmax only. Temperature/top-p sampling not yet implemented.

## Future Work

- [ ] Implement **Q4_0** and **Q4_K** dequantization (crucial for running 1B+ models)
- [ ] Add temperature and top-p sampling
- [ ] Interactive chat mode with conversation history
- [ ] KV-Cache optimization for faster long-context generation
- [ ] HTTP API server for remote inference
- [ ] Support for more model architectures (Phi, Qwen, etc.)

## Project Structure

```
vc4llm/
├── vc4llm.cpp          # Main source file (single-file implementation)
├── README.md           # This file
├── LICENSE             # GPL-3.0 license
└── .gitignore          # Git ignore rules
```

## Contributing

Contributions are welcome! Areas where help is needed:

- Q4_K quantization support
- Additional model architecture support
- Performance optimizations
- Testing on other Raspberry Pi models

Please open an issue first to discuss proposed changes.

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [**llama.cpp**](https://github.com/ggerganov/llama.cpp): The inspiration for this project and the creator of the GGUF format
- [**VC4CL**](https://github.com/doe300/VC4CL): The open-source OpenCL implementation for the Raspberry Pi VideoCore IV GPU
- [**HuggingFace**](https://huggingface.co/): For hosting the SmolLM and other small language models
