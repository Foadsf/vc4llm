# VC4LLM: VideoCore IV LLM Inference Engine

**VC4LLM** is a lightweight, bare-metal-style Large Language Model (LLM) inference engine built from scratch for the **Raspberry Pi 3 Model B**. It is designed to squeeze maximum performance out of the Cortex-A53 CPU using NEON SIMD and explore the capabilities of the VideoCore IV GPU via VC4CL OpenCL.

## 📖 Overview

Running modern LLMs on older edge hardware like the Raspberry Pi 3B (1GB RAM, quad-core A53) is a significant challenge. Existing frameworks often have high overhead or lack specific optimizations for this architecture.

**VC4LLM** solves this by:

1. **Zero Dependencies**: No external BLAS, Torch, or Python runtime required. Just C++17.
2. **Hardware Optimization**: Custom NEON assembly-level intrinsics for matrix multiplication.
3. **Experimental GPU Support**: Direct targeting of the VideoCore IV QPU for compute.

This project serves as both a practical inference tool for small models (like SmolLM-135M) and a research platform for low-level optimization on the Raspberry Pi.

## ✨ Features

- **GGUF v3 Support**: Natively loads modern GGUF model files.
- **NEON SIMD Acceleration**: Hand-tuned kernels for Q8_0 dot products and vector operations.
- **Multi-threading**: Custom lightweight thread pool for parallelizing matrix operations.
- **Memory Efficiency**: Uses `mmap` for instant model loading and OS-managed paging.
- **BPE Tokenizer**: Full byte-level Byte Pair Encoding (GPT-2 style) support.
- **Tied Embeddings**: Support for small models that share input/output embedding weights.
- **Hybrid Compute**: Experimental support for offloading layers to the VideoCore IV GPU via OpenCL.

## 🚀 Performance Benchmarks

**Hardware:** Raspberry Pi 3 Model B v1.2 (Quad-core Cortex-A53 @ 1.2GHz, 1GB RAM)  
**Model:** SmolLM2-135M-Instruct-Q8_0.gguf

| Configuration | Speed (tok/s) | Notes |
| :--- | :--- | :--- |
| Phase 1 (Scalar CPU) | 0.54 | Baseline naive C++ implementation. |
| **Phase 2 (NEON + 4 Threads)** | **5.46** | **~10x Speedup.** SIMD + Parallelism. |
| Phase 3 (GPU OpenCL) | 0.30 | Experimental. Limited by memory bandwidth & kernel overhead. |

> **Note**: The CPU path is currently the fastest and recommended way to run models on the Pi 3B.

## 🛠 Requirements

### Hardware

- **Raspberry Pi 3 Model B / B+** (or Pi 2 v1.2 with Cortex-A53)
- **MicroSD Card**: Class 10 or UHS-I recommended for fast paging.
- **Power Supply**: Reliable 5V 2.5A supply (critical for max CPU/GPU load).

### Software

- **OS**: Raspberry Pi OS (Legacy) **32-bit** (Bookworm or Bullseye).
- **Compiler**: GCC 8+ (needs C++17 support).
- **(Optional) GPU Drivers**: VC4CL OpenCL implementation for VideoCore IV.

## 🏗 Building

1. **Clone or Create Source:**  
   Save the `vc4llm.cpp` file to your Pi.

2. **Install Dependencies:**

   ```bash
   sudo apt update
   sudo apt install build-essential git
   # Optional: OpenCL headers if testing GPU
   sudo apt install ocl-icd-opencl-dev
   ```

3. **Compile (CPU Optimized - Recommended):**

   ```bash
   g++ -O3 -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard \
       -o vc4llm vc4llm.cpp -lpthread
   ```

4. **Compile (With GPU Support):**

   ```bash
   g++ -O3 -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard \
       -o vc4llm vc4llm.cpp -lpthread -lOpenCL
   ```

## 💻 Usage

### Basic Inference (CPU)

Run a model with a prompt:

```bash
./vc4llm -m models/SmolLM2-135M.gguf -p "Once upon a time" -n 50
```

### Full Options

```
./vc4llm [flags]

Flags:
  -m <file>    Path to GGUF model file (Required)
  -p <text>    Input prompt (Default: "Hello world")
  -n <int>     Number of tokens to predict (Default: 20)
  -t <int>     Number of CPU threads (Default: 4)
  -v           Verbose output (print layer info)
  --gpu        Enable experimental OpenCL GPU acceleration
```

### Example Output

```
Model mmapped: SmolLM2-135M-Instruct-Q8_0.gguf (138 MB)
Inference init: dim=576, hidden=1536, head_dim=64, threads=4, gpu=OFF
Prompt: 'Once upon a time' (5 tokens)
Generating 20 tokens...
 there was a little girl named Lily. She loved to explore the forest near her house. One day,
Done. Time: 3.66s (5.46 tok/s)
```

## 📦 Model Compatibility

VC4LLM supports **GGUF v3** models. Due to the 1GB RAM limit, you are restricted to small models (typically <500M parameters).

**Tested & Recommended Models:**

- **SmolLM2-135M-Instruct** (Q8_0 or Q4_K_M)
- **TinyLlama-1.1B** (Very tight fit, requires swap)

**Where to get models:**  
Search HuggingFace for "GGUF". Example download:

```bash
wget https://huggingface.co/lmstudio-community/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf
```

## 🧠 Lessons Learned & Technical Details

Developing this engine revealed several critical insights about low-level programming on the Raspberry Pi:

### 1. GGUF Parsing & Alignment

- **Bus Errors**: The Pi's ARM processor strictly enforces memory alignment. Casting a raw byte pointer `uint8_t*` to `uint64_t*` causes a `SIGBUS` if the address isn't 8-byte aligned.  
  **Fix**: Always use `memcpy()` to read multi-byte values from a raw buffer.

- **Hidden Fields**: GGUF arrays have a hidden `element_type` field before the count. Missing this leads to parsing garbage values for array sizes.

### 2. NEON Optimization (The 10x Boost)

- **Scalar vs SIMD**: Naive C++ loops are incredibly slow for matrix multiplication. Using `vld1q`, `vmulq`, and `vmlaq` NEON intrinsics provided a 10x speedup.

- **32-bit Limitations**: Many modern NEON tutorials assume AArch64 (64-bit). The Pi 3B running 32-bit OS lacks instructions like `vaddvq_f32` (vector add across all lanes). We had to implement manual horizontal addition using `vpadd_f32`.

### 3. GPU Acceleration (VC4CL)

- **Instruction Support**: The VideoCore IV GPU via VC4CL is extremely limited. It lacks standard intrinsics like `llvm.ctlz` (count leading zeros), meaning complex FP16 conversions fail to compile. We had to write a simplified math-only converter.

- **Vector Widths**: The `cl_arm_integer_dot_product_int8` extension is strict. `arm_dot` takes `char4` (not `uchar4` or generic pointers).

- **Bottlenecks**: Without persistent buffers, the overhead of `clCreateBuffer` (mapping memory) destroys performance. Even with optimizations, the GPU's memory bandwidth and lack of caching make it slower than the CPU for linear inference of small batches.

## ⚠️ Known Limitations

1. **Quantization Support**: Currently only supports **F32** and **Q8_0** (8-bit) weights. Q4_K support is planned.
2. **Context Length**: Fixed at model default (usually 2048 or 8192), but practical generation limit is RAM.
3. **GPU Performance**: The GPU path is functional but currently slower than the CPU. Use CPU mode for best results.
4. **Platform**: Heavily optimized for ARMv8. May not compile on x86 without modifications to SIMD code.

## 🔮 Future Work

- [ ] Implement **Q4_0** and **Q4_K** dequantization (crucial for running 1B+ models).
- [ ] Optimize GPU kernel to use Q8_0 weights with F32 input without requantization (hybrid kernel).
- [ ] Add KV-Cache to speed up long context generation.
- [ ] Interactive chat mode.

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [**llama.cpp**](https://github.com/ggerganov/llama.cpp): The inspiration for this project and the creator of the GGUF format.
- [**VC4CL**](https://github.com/doe300/VC4CL): The open-source OpenCL implementation for the Raspberry Pi VideoCore IV GPU.
- **HuggingFace**: For hosting the amazing SmolLM models.
