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
3. **Experimental GPU Support**: Direct targeting of the VideoCore IV QPU for compute.

This project serves as both a practical inference tool for small models (like SmolLM-135M) and a research platform for low-level optimization on the Raspberry Pi.

## Features

- **GGUF v3 Support**: Natively loads modern GGUF model files
- **NEON SIMD Acceleration**: Hand-tuned kernels for Q8_0 dot products and vector operations
- **Multi-threading**: Custom lightweight thread pool for parallelizing matrix operations
- **Memory Efficiency**: Uses `mmap` for instant model loading and OS-managed paging
- **BPE Tokenizer**: Full byte-level Byte Pair Encoding (GPT-2 style) support
- **Tied Embeddings**: Support for small models that share input/output embedding weights
- **Hybrid Compute**: Experimental support for offloading layers to the VideoCore IV GPU via OpenCL

## Performance Benchmarks

**Hardware**: Raspberry Pi 3 Model B v1.2 (Quad-core Cortex-A53 @ 1.2GHz, 1GB RAM)  
**Model**: SmolLM2-135M-Instruct-Q8_0.gguf (138 MB)

| Configuration | Speed (tok/s) | Notes |
|:--------------|:-------------:|:------|
| Phase 1 (Scalar CPU) | 0.54 | Baseline naive C++ implementation |
| **Phase 2 (NEON + 4 Threads)** | **5.46** | **~10x Speedup** - SIMD + Parallelism |
| Phase 3 (GPU OpenCL) | 0.30 | Experimental - Limited by memory bandwidth |

> **Note**: The CPU path is currently the fastest and recommended way to run models on the Pi 3B.

## Requirements

### Hardware

- **Raspberry Pi 3 Model B / B+** (or Pi 2 v1.2 with Cortex-A53)
- **MicroSD Card**: Class 10 or UHS-I recommended for fast paging
- **Power Supply**: Reliable 5V 2.5A supply (critical for max CPU/GPU load)

### Software

- **OS**: Raspberry Pi OS (Legacy) **32-bit** (Bookworm or Bullseye)
- **Compiler**: GCC 8+ (needs C++17 support)
- **(Optional)** VC4CL OpenCL implementation for VideoCore IV GPU acceleration

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

# Optional: OpenCL headers for GPU support
sudo apt install ocl-icd-opencl-dev
```

### 3. Compile (CPU Optimized - Recommended)

```bash
g++ -O3 -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard \
    -o vc4llm vc4llm.cpp -lpthread
```

### 4. Compile (With GPU Support)

```bash
g++ -O3 -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard \
    -o vc4llm vc4llm.cpp -lpthread -lOpenCL
```

## Usage

### Basic Inference

```bash
./vc4llm -m SmolLM2-135M-Instruct-Q8_0.gguf -p "Once upon a time" -n 50
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
  --gpu        Enable experimental OpenCL GPU acceleration (requires sudo)
```

### Example Output

```
$ ./vc4llm -m SmolLM2-135M-Instruct-Q8_0.gguf -p "Once upon a time" -n 20 -t 4

Model mmapped: SmolLM2-135M-Instruct-Q8_0.gguf (138 MB)
Prompt: 'Once upon a time' (5 tokens)
Generating 20 tokens with 4 threads...
Inference init: dim=576, hidden=1536, head_dim=64, threads=4, gpu=OFF
You are a skilled and cunning pirate who has been sailing the seven seas for many years. You
Done. Time: 3.66s (5.46 tok/s)
```

### GPU Mode (Experimental)

```bash
# GPU mode requires root access for /dev/mem
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

## Lessons Learned

Developing this engine revealed several critical insights about low-level programming on the Raspberry Pi:

### 1. GGUF Parsing & Memory Alignment

- **Bus Errors**: The Pi's ARM processor strictly enforces memory alignment. Casting a raw byte pointer (`uint8_t*`) to `uint64_t*` causes a `SIGBUS` if the address isn't 8-byte aligned.
  
  **Fix**: Always use `memcpy()` to read multi-byte values from raw buffers:
  ```cpp
  // WRONG - causes SIGBUS on unaligned addresses
  uint64_t val = *reinterpret_cast<uint64_t*>(ptr);
  
  // CORRECT - works on any alignment
  uint64_t val;
  memcpy(&val, ptr, sizeof(val));
  ```

- **Hidden Fields**: GGUF v3 arrays have a hidden `element_type` field before the count. Missing this leads to parsing garbage values for array sizes.

### 2. NEON Optimization (The 10x Boost)

- **Scalar vs SIMD**: Naive C++ loops are incredibly slow for matrix multiplication. Using `vld1q`, `vmulq`, and `vmlaq` NEON intrinsics provided a 10x speedup.

- **32-bit ARM Limitations**: Many modern NEON tutorials assume AArch64 (64-bit). The Pi 3B running 32-bit Raspberry Pi OS lacks instructions like `vaddvq_f32` (horizontal vector sum).
  
  **Fix**: Manual horizontal addition for 32-bit ARM:
  ```cpp
  // AArch64 only (doesn't work on 32-bit Pi OS)
  float sum = vaddvq_f32(vec);
  
  // 32-bit ARM compatible
  float32x2_t pair = vadd_f32(vget_low_f32(vec), vget_high_f32(vec));
  pair = vpadd_f32(pair, pair);
  float sum = vget_lane_f32(pair, 0);
  ```

### 3. GPU Acceleration (VC4CL)

- **Instruction Support**: The VideoCore IV GPU via VC4CL is extremely limited. It lacks standard intrinsics like `llvm.ctlz` (count leading zeros), meaning complex FP16 conversions with denormal handling fail to compile.
  
  **Fix**: Simplified f16→f32 that treats denormals as zero (acceptable for model weights).

- **Integer Dot Product**: The `cl_arm_integer_dot_product_int8` extension's `arm_dot()` takes exactly 2 arguments (`char4`, `char4`), not 3. Many tutorials show incorrect usage.

- **Memory Bottlenecks**: Without persistent buffers, the overhead of `clCreateBuffer` (mapping memory) on every operation destroys performance. Even with optimizations, the GPU's limited memory bandwidth makes it slower than the CPU for single-token inference.

### 4. ThreadPool Synchronization

- **Condition Variable Pitfalls**: Standard condition variable patterns can deadlock if notifications fire before the main thread starts waiting.
  
  **Fix**: Use spin-wait with `std::atomic` for simple barrier synchronization, or carefully designed CV patterns with proper mutex scoping.

## Known Limitations

1. **Quantization Support**: Currently only supports **F32** and **Q8_0** weights. Q4_K support is planned.
2. **Context Length**: Fixed at model default (usually 2048), practical limit depends on available RAM.
3. **GPU Performance**: The GPU path is functional but currently slower than CPU. Use CPU mode for best results.
4. **Platform**: Heavily optimized for 32-bit ARMv8. Requires modifications for x86 or 64-bit ARM.
5. **Sampling**: Currently uses greedy argmax only. Temperature/top-p sampling not yet implemented.

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
- [**Anthropic Claude**](https://www.anthropic.com/) and [**Google Gemini**](https://deepmind.google/technologies/gemini/): AI assistants that helped develop and debug this project

## References

- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [VC4CL Documentation](https://github.com/doe300/VC4CL/wiki)
- [LLaMA Architecture Paper](https://arxiv.org/abs/2302.13971)
