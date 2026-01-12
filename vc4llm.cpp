/*
 * VC4LLM: VideoCore IV LLM Inference Engine
 * Phase 3: GPU Acceleration (VC4CL OpenCL) - Optimized
 *
 * Target Hardware: Raspberry Pi 3B (Cortex-A53 + VideoCore IV)
 * Features:
 * - GGUF v3 Loader (mmap)
 * - NEON-optimized CPU Inference (Q8_0/F32)
 * - OpenCL-accelerated GPU Inference (Q8_0) via VC4CL
 * - Multi-threaded CPU Fallback
 *
 * Compile: g++ -O3 -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard -o vc4llm vc4llm.cpp -lpthread -lOpenCL
 */

#if __has_include(<CL/cl.h>)
    #define CL_TARGET_OPENCL_VERSION 120
    #include <CL/cl.h>
    #define HAS_OPENCL 1
#else
    #define HAS_OPENCL 0
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iomanip>
#include <set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <atomic>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

// =================================================================================================
// 1. Constants & Types
// =================================================================================================

#define GGUF_MAGIC 0x46554747 // "GGUF"
#define GGUF_VERSION 3

// GGML Types
enum ggml_type : uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 11,
    GGML_TYPE_Q3_K = 12,
    GGML_TYPE_Q4_K = 15,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 16,
    GGML_TYPE_COUNT,
};

#define Q8_0_BLOCK_SIZE 32

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
};

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t ne[4];
    ggml_type type;
    uint64_t offset;
    void* data = nullptr;
    
    uint64_t num_elements() const {
        uint64_t n = 1;
        for(uint32_t i=0; i<n_dims; ++i) n *= ne[i];
        return n;
    }

    size_t size_bytes() const {
        uint64_t n = num_elements();
        switch(type) {
            case GGML_TYPE_F32: return n * 4;
            case GGML_TYPE_F16: return n * 2;
            case GGML_TYPE_Q8_0: return (n / 32) * 34;
            case GGML_TYPE_Q4_0: return (n / 32) * 18;
            default: return n;
        }
    }
};

struct ModelConfig {
    uint32_t n_embd = 0;
    uint32_t n_layer = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_ctx = 512;
    float rope_freq_base = 10000.0f;
    float rms_norm_eps = 1e-5f;
    uint32_t vocab_size = 0;
};

struct TensorNaming {
    std::string token_embd;
    std::string output_norm;
    std::string output;
    std::string layer_prefix; 
    std::string attn_norm;
    std::string attn_q;
    std::string attn_k;
    std::string attn_v;
    std::string attn_out;
    std::string ffn_norm;
    std::string ffn_gate;
    std::string ffn_up;
    std::string ffn_down;
};

// =================================================================================================
// 2. Utils & ThreadPool
// =================================================================================================

uint64_t time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 0x1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    if (e == 0) return (m == 0) ? (s ? -0.0f : 0.0f) : (s ? -1.0f : 1.0f) * std::ldexp((float)m, -24);
    if (e == 31) return (s ? -1.0f : 1.0f) * (m ? NAN : INFINITY);
    return (s ? -1.0f : 1.0f) * std::ldexp((float)(m + 1024), e - 25);
}

// Safe unaligned read helpers
template<typename T>
T read_unaligned(uint8_t*& ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    ptr += sizeof(T);
    return val;
}

template<typename T>
T peek_unaligned(const void* ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    return val;
}

class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& worker : workers) worker.join();
    }

    // Simple, correct parallel_for using spin-wait
    void parallel_for(int start, int end, std::function<void(int, int)> body) {
        if (start >= end) return;

        int n_threads = workers.size();
        int range = end - start;
        int chunk = (range + n_threads - 1) / n_threads;
        
        std::atomic<int> remaining(0);

        for (int t = 0; t < n_threads && start + t * chunk < end; ++t) {
            remaining++;
            int t_start = start + t * chunk;
            int t_end = std::min(t_start + chunk, end);
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                tasks.emplace([=, &body, &remaining] {
                    body(t_start, t_end);
                    remaining.fetch_sub(1, std::memory_order_release);
                });
            }
            condition.notify_one();
        }

        // Spin-wait (simple, no deadlock possible)
        while (remaining.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// =================================================================================================
// 3. OpenCL Backend (VC4CL)
// =================================================================================================

#if HAS_OPENCL
static cl_context g_cl_context = nullptr;
static cl_command_queue g_cl_queue = nullptr;
static cl_program g_cl_program = nullptr;
static cl_kernel g_cl_kernel_q8_0 = nullptr;
static bool g_cl_initialized = false;
#endif

static bool g_use_gpu = false;

struct PerfCounters {
    #if HAS_OPENCL
    uint64_t gpu_kernel_us = 0;
    uint64_t gpu_transfer_us = 0;
    int gpu_calls = 0;
    #endif
    uint64_t cpu_compute_us = 0;
    int cpu_calls = 0;

    void report() {
        std::cout << "\n=== Performance Breakdown ===" << std::endl;
        #if HAS_OPENCL
        std::cout << "GPU kernel time: " << gpu_kernel_us / 1000.0 << " ms" << std::endl;
        std::cout << "GPU transfer time: " << gpu_transfer_us / 1000.0 << " ms" << std::endl;
        std::cout << "GPU calls: " << gpu_calls << std::endl;
        #endif
        std::cout << "CPU compute time: " << cpu_compute_us / 1000.0 << " ms" << std::endl;
        std::cout << "CPU calls: " << cpu_calls << std::endl;
    }
};

static PerfCounters g_perf;

#if HAS_OPENCL
// OpenCL Kernel Source
const char* CL_KERNEL_SRC = R"(
// FP16 to FP32 conversion (simplified, no denormal handling)
float f16_to_f32(ushort h) {
    uint s = (h >> 15) & 0x1;
    uint e = (h >> 10) & 0x1F;
    uint m = h & 0x3FF;
    if (e == 0) return 0.0f;
    if (e == 31) return s ? -INFINITY : INFINITY;
    uint val = (s << 31) | ((e + 112) << 23) | (m << 13);
    return as_float(val);
}

// Optimized GEMV kernel for Q8_0 weights × F32 input
// Strategy: Each work item processes one output row
// Use float4 vectorization where possible
__kernel void gemv_q8_0_opt(
    __global const uchar* restrict W,  // Q8_0 weights [M, K/32, 34]
    __global const float* restrict x,  // Float input [K]
    __global float* restrict y,        // Output [M]
    const int M,
    const int K
) {
    const int row = get_global_id(0);
    if (row >= M) return;
    
    const int num_blocks = K >> 5;  // K / 32
    float total = 0.0f;
    
    // Pointer to this row's weights
    __global const uchar* w_ptr = W + row * num_blocks * 34;
    
    for (int b = 0; b < num_blocks; b++) {
        // Load scale
        const float d = f16_to_f32(*(__global const ushort*)w_ptr);
        __global const char* qs = (__global const char*)(w_ptr + 2);
        
        // Vectorized accumulation using float4
        // Process 32 elements as 8 × float4
        float4 sum4 = (float4)(0.0f);

        __global const float* x_ptr = x + (b << 5);  // b * 32

        // Unroll 8 iterations of 4 elements each
        for (int j = 0; j < 32; j += 4) {
            char4 w4 = vload4(0, qs + j);
            float4 x4 = vload4(0, x_ptr + j);
            float4 w_f = convert_float4(w4);
            sum4 = mad(w_f, x4, sum4);  // fused multiply-add
        }

        // Horizontal sum of float4
        float block_sum = sum4.x + sum4.y + sum4.z + sum4.w;
        total += block_sum * d;

        w_ptr += 34;
    }

    y[row] = total;
}
)";

bool init_opencl() {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    
    // Get VC4CL platform
    if (clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS) {
        std::cerr << "OpenCL: No platform found" << std::endl;
        return false;
    }
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) != CL_SUCCESS) {
        std::cerr << "OpenCL: No GPU found" << std::endl;
        return false;
    }
    
    char name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    std::cout << "OpenCL Device: " << name << std::endl;
    
    g_cl_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!g_cl_context) return false;
    
    g_cl_queue = clCreateCommandQueue(g_cl_context, device, 0, &err);
    if (!g_cl_queue) return false;
    
    // Check for required extensions
    char extensions[4096];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);

    bool has_int8_dot = (strstr(extensions, "cl_arm_integer_dot_product_int8") != nullptr);
    std::cout << "INT8 dot product: " << (has_int8_dot ? "YES" : "NO") << std::endl;

    // Compile with appropriate options
    const char* build_opts = has_int8_dot ?
        "-cl-fast-relaxed-math -DUSE_INT8_DOT=1" :
        "-cl-fast-relaxed-math -DUSE_INT8_DOT=0";

    // Compile program
    const char* src_ptr = CL_KERNEL_SRC;
    g_cl_program = clCreateProgramWithSource(g_cl_context, 1, &src_ptr, NULL, &err);
    if (clBuildProgram(g_cl_program, 1, &device, build_opts, NULL, NULL) != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(g_cl_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        std::vector<char> log(len);
        clGetProgramBuildInfo(g_cl_program, device, CL_PROGRAM_BUILD_LOG, len, log.data(), NULL);
        std::cerr << "OpenCL Build Error:\n" << log.data() << std::endl;
        return false;
    }
    
    g_cl_kernel_q8_0 = clCreateKernel(g_cl_program, "gemv_q8_0_opt", &err);
    if (err != CL_SUCCESS) return false;
    
    g_cl_initialized = true;
    return true;
}

bool gpu_gemv(float* dst, const float* x, cl_mem w_buf, int m, int k) {
    cl_int err;
    auto t0 = time_us();

    // Reusable input buffer (create once, resize if needed)
    static cl_mem x_buf = nullptr;
    static cl_mem y_buf = nullptr;
    static int x_buf_size = 0;
    static int y_buf_size = 0;

    // Resize input buffer if needed
    if (k * sizeof(float) > x_buf_size) {
        if (x_buf) clReleaseMemObject(x_buf);
        x_buf = clCreateBuffer(g_cl_context, CL_MEM_READ_ONLY,
                               k * sizeof(float), NULL, &err);
        x_buf_size = k * sizeof(float);
    }

    // Resize output buffer if needed
    if (m * sizeof(float) > y_buf_size) {
        if (y_buf) clReleaseMemObject(y_buf);
        y_buf = clCreateBuffer(g_cl_context, CL_MEM_WRITE_ONLY,
                               m * sizeof(float), NULL, &err);
        y_buf_size = m * sizeof(float);
    }

    auto t1 = time_us();
    g_perf.gpu_transfer_us += (t1 - t0);

    // Async write input
    err = clEnqueueWriteBuffer(g_cl_queue, x_buf, CL_FALSE, 0,
                               k * sizeof(float), x, 0, NULL, NULL);
    if (err != CL_SUCCESS) return false;

    auto t2 = time_us();
    g_perf.gpu_transfer_us += (t2 - t1);

    // Set kernel args
    clSetKernelArg(g_cl_kernel_q8_0, 0, sizeof(cl_mem), &w_buf);
    clSetKernelArg(g_cl_kernel_q8_0, 1, sizeof(cl_mem), &x_buf);
    clSetKernelArg(g_cl_kernel_q8_0, 2, sizeof(cl_mem), &y_buf);
    clSetKernelArg(g_cl_kernel_q8_0, 3, sizeof(int), &m);
    clSetKernelArg(g_cl_kernel_q8_0, 4, sizeof(int), &k);

    // Launch kernel
    size_t global_size = m;
    size_t local_size = std::min(12, m);  // VC4CL max is 12

    err = clEnqueueNDRangeKernel(g_cl_queue, g_cl_kernel_q8_0, 1, NULL,
                                  &global_size, &local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) return false;

    auto t3 = time_us();
    g_perf.gpu_kernel_us += (t3 - t2);

    // Blocking read output
    err = clEnqueueReadBuffer(g_cl_queue, y_buf, CL_TRUE, 0,
                              m * sizeof(float), dst, 0, NULL, NULL);

    auto t4 = time_us();
    g_perf.gpu_transfer_us += (t4 - t3);

    return (err == CL_SUCCESS);
}
#endif

// =================================================================================================
// 4. GGUF Parser
// =================================================================================================

class GGUFModel {
public:
    ModelConfig config;
    TensorNaming naming;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, TensorInfo*> tensor_map;
    
    void* mapped_data = nullptr;
    size_t mapped_size = 0;
    
    std::vector<std::string> vocab_tokens;
    std::vector<float> vocab_scores;
    std::map<std::string, int> token_to_id;
    int token_bos = 1, token_eos = 2;

    #if HAS_OPENCL
    // Add GPU buffer cache
    std::unordered_map<std::string, cl_mem> gpu_buffers;
    #endif

    ~GGUFModel() {
        #if HAS_OPENCL
        for (auto& kv : gpu_buffers) {
            clReleaseMemObject(kv.second);
        }
        gpu_buffers.clear();

        if (g_cl_initialized) {
            clReleaseKernel(g_cl_kernel_q8_0);
            clReleaseProgram(g_cl_program);
            clReleaseCommandQueue(g_cl_queue);
            clReleaseContext(g_cl_context);
        }
        #endif
        if (mapped_data) munmap(mapped_data, mapped_size);
    }

    bool load(const std::string& path, bool verbose = false) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) return false;
        struct stat sb; fstat(fd, &sb);
        mapped_size = sb.st_size;
        mapped_data = mmap(NULL, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (mapped_data == MAP_FAILED) return false;

        std::cout << "Model mmapped: " << path << " (" << mapped_size / 1024 / 1024 << " MB)" << std::endl;

        uint8_t* ptr = static_cast<uint8_t*>(mapped_data);
        
        GGUFHeader header;
        header.magic = read_unaligned<uint32_t>(ptr);
        header.version = read_unaligned<uint32_t>(ptr);
        header.tensor_count = read_unaligned<uint64_t>(ptr);
        header.kv_count = read_unaligned<uint64_t>(ptr);

        if (header.magic != GGUF_MAGIC || header.version != GGUF_VERSION) return false;

        for (uint64_t i = 0; i < header.kv_count; ++i) {
            std::string key = read_string(ptr);
            uint32_t type = read_unaligned<uint32_t>(ptr);
            if (verbose) std::cout << "  KV: " << key << std::endl;
            
            if (key == "llama.embedding_length" || key == "gpt2.embedding_length") config.n_embd = read_val<uint32_t>(ptr, type);
            else if (key == "llama.block_count" || key == "gpt2.block_count") config.n_layer = read_val<uint32_t>(ptr, type);
            else if (key == "llama.attention.head_count" || key == "gpt2.attention.head_count") config.n_head = read_val<uint32_t>(ptr, type);
            else if (key == "llama.attention.head_count_kv") config.n_head_kv = read_val<uint32_t>(ptr, type);
            else if (key == "llama.context_length" || key == "gpt2.context_length") config.n_ctx = read_val<uint32_t>(ptr, type);
            else if (key == "llama.rope.freq_base") config.rope_freq_base = read_val<float>(ptr, type);
            else if (key == "tokenizer.ggml.bos_token_id") token_bos = read_val<uint32_t>(ptr, type);
            else if (key == "tokenizer.ggml.eos_token_id") token_eos = read_val<uint32_t>(ptr, type);
            else if (key == "tokenizer.ggml.tokens") {
                uint32_t item_type = read_unaligned<uint32_t>(ptr); 
                uint64_t count = read_unaligned<uint64_t>(ptr);
                config.vocab_size = count;
                vocab_tokens.resize(count);
                for(uint64_t j=0; j<count; ++j) {
                    vocab_tokens[j] = read_string(ptr);
                    token_to_id[vocab_tokens[j]] = j;
                }
            }
            else if (key == "tokenizer.ggml.scores") {
                uint32_t item_type = read_unaligned<uint32_t>(ptr);
                uint64_t count = read_unaligned<uint64_t>(ptr);
                vocab_scores.resize(count);
                for(uint64_t j=0; j<count; ++j) vocab_scores[j] = read_unaligned<float>(ptr);
            }
            else skip_val(ptr, type);
        }

        if (config.n_head_kv == 0) config.n_head_kv = config.n_head;

        for (uint64_t i = 0; i < header.tensor_count; ++i) {
            TensorInfo info;
            info.name = read_string(ptr);
            info.n_dims = read_unaligned<uint32_t>(ptr);
            for(uint32_t d=0; d<info.n_dims; ++d) info.ne[d] = read_unaligned<uint64_t>(ptr);
            info.type = (ggml_type)read_unaligned<uint32_t>(ptr);
            info.offset = read_unaligned<uint64_t>(ptr);
            tensors.push_back(info);
        }

        uintptr_t ptr_int = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t start_int = reinterpret_cast<uintptr_t>(mapped_data);
        uint64_t offset_base = (ptr_int - start_int + 63) & ~63;

        for (auto& t : tensors) {
            t.data = static_cast<char*>(mapped_data) + offset_base + t.offset;
            tensor_map[t.name] = &t;
        }

        discover_names();

        #if HAS_OPENCL
        // After tensors are loaded, upload Q8_0 weights to GPU
        if (g_use_gpu && g_cl_initialized) {
            upload_weights_to_gpu();
        }
        #endif

        return true;
    }

    TensorInfo* get_tensor(const std::string& name) {
        if (tensor_map.find(name) != tensor_map.end()) return tensor_map[name];
        return nullptr;
    }

    #if HAS_OPENCL
    void upload_weights_to_gpu() {
        std::cout << "Uploading weights to GPU..." << std::endl;
        for (auto& t : tensors) {
            if (t.type == GGML_TYPE_Q8_0) {
                cl_int err;
                // USE_HOST_PTR for zero-copy from mmap
                cl_mem buf = clCreateBuffer(
                    g_cl_context,
                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                    t.size_bytes(),
                    t.data,
                    &err
                );
                if (err == CL_SUCCESS) {
                    gpu_buffers[t.name] = buf;
                }
            }
        }
        std::cout << "Uploaded " << gpu_buffers.size() << " tensors to GPU" << std::endl;
    }
    #endif

private:
    void discover_names() {
        bool has_blk = (tensor_map.find("blk.0.attn_q.weight") != tensor_map.end());
        bool has_model = (tensor_map.find("model.layers.0.self_attn.q_proj.weight") != tensor_map.end());
        
        if (has_blk) {
            naming = {"token_embd.weight", "output_norm.weight", "output.weight", "blk.%d.", 
                      "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                      "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
            if (tensor_map.find("output.weight") == tensor_map.end()) naming.output = "token_embd.weight"; 
        } else if (has_model) {
            naming = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "model.layers.%d.",
                      "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                      "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"};
            if (tensor_map.find("lm_head.weight") == tensor_map.end()) naming.output = "model.embed_tokens.weight";
        } else {
            naming.token_embd = "token_embd.weight"; naming.layer_prefix = "blk.%d."; 
        }
    }

    std::string read_string(uint8_t*& ptr) {
        uint64_t len = read_unaligned<uint64_t>(ptr);
        std::string s(reinterpret_cast<char*>(ptr), len);
        ptr += len;
        return s;
    }

    template<typename T>
    T read_val(uint8_t*& ptr, uint32_t type) {
        T val = 0;
        switch(type) {
            case 2: val = (T)read_unaligned<uint16_t>(ptr); break;
            case 3: val = (T)read_unaligned<int16_t>(ptr); break;
            case 4: val = (T)read_unaligned<uint32_t>(ptr); break;
            case 5: val = (T)read_unaligned<int32_t>(ptr); break;
            case 6: val = (T)read_unaligned<float>(ptr); break;
            case 7: val = (T)read_unaligned<bool>(ptr); break;
            case 10: val = (T)read_unaligned<uint64_t>(ptr); break;
            case 11: val = (T)read_unaligned<int64_t>(ptr); break;
        }
        return val;
    }

    void skip_val(uint8_t*& ptr, uint32_t type) {
        switch(type) {
            case 0: case 1: case 7: ptr += 1; break;
            case 2: case 3: ptr += 2; break;
            case 4: case 5: case 6: ptr += 4; break;
            case 8: ptr += read_unaligned<uint64_t>(ptr); break;
            case 9: {
                uint32_t itype = read_unaligned<uint32_t>(ptr);
                uint64_t len = read_unaligned<uint64_t>(ptr);
                for(uint64_t i=0; i<len; ++i) skip_val(ptr, itype);
                break;
            }
            case 10: case 11: ptr += 8; break;
        }
    }
};

// =================================================================================================
// 5. Tokenizer & BPE
// =================================================================================================

std::vector<int> tokenize(GGUFModel& model, const std::string& text) {
    std::vector<int> tokens;
    if (model.token_bos != -1) tokens.push_back(model.token_bos);
    std::string input = text;
    size_t i = 0;
    while (i < input.length()) {
        int best_id = -1;
        size_t best_len = 0;
        for (size_t len = 1; len <= input.length() - i; ++len) {
            std::string sub = input.substr(i, len);
            if (model.token_to_id.count(sub)) {
                best_len = len;
                best_id = model.token_to_id[sub];
            }
        }
        if (best_id == -1) { i++; continue; }
        tokens.push_back(best_id);
        i += best_len;
    }
    return tokens;
}

std::unordered_map<char32_t, uint8_t> build_byte_decoder() {
    std::unordered_map<char32_t, uint8_t> decoder;
    for (int b = 33; b <= 126; b++) decoder[b] = b;
    for (int b = 161; b <= 172; b++) decoder[b] = b;
    for (int b = 174; b <= 255; b++) decoder[b] = b;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) continue;
        decoder[256 + n++] = b;
    }
    return decoder;
}

char32_t decode_utf8_char(const char*& ptr) {
    uint8_t c = *ptr++;
    if ((c & 0x80) == 0) return c;
    if ((c & 0xE0) == 0xC0) { char32_t cp = (c & 0x1F) << 6; cp |= (*ptr++ & 0x3F); return cp; }
    if ((c & 0xF0) == 0xE0) { char32_t cp = (c & 0x0F) << 12; cp |= (*ptr++ & 0x3F) << 6; cp |= (*ptr++ & 0x3F); return cp; }
    if ((c & 0xF8) == 0xF0) { char32_t cp = (c & 0x07) << 18; cp |= (*ptr++ & 0x3F) << 12; cp |= (*ptr++ & 0x3F) << 6; cp |= (*ptr++ & 0x3F); return cp; }
    return 0xFFFD;
}

static std::unordered_map<char32_t, uint8_t> g_byte_decoder;
static bool g_byte_decoder_initialized = false;

std::string detokenize(GGUFModel& model, int id) {
    if (!g_byte_decoder_initialized) { g_byte_decoder = build_byte_decoder(); g_byte_decoder_initialized = true; }
    if (id < 0 || id >= model.vocab_tokens.size()) return "";
    const std::string& token = model.vocab_tokens[id];
    if (token == "<s>" || token == "</s>" || token == "<unk>" || token == "<pad>" || token == "<|endoftext|>") return "";
    std::string result;
    const char* ptr = token.c_str();
    const char* end = ptr + token.length();
    while (ptr < end) {
        char32_t cp = decode_utf8_char(ptr);
        auto it = g_byte_decoder.find(cp);
        if (it != g_byte_decoder.end()) result += (char)it->second;
        else if (cp < 0x80) result += (char)cp; 
    }
    return result;
}

// =================================================================================================
// 6. Compute Primitives (CPU & GPU)
// =================================================================================================

static std::set<int> unsupported_types_warned;

float get_tensor_f32(TensorInfo* t, int i) {
    if (t->type == GGML_TYPE_F32) return ((float*)t->data)[i];
    if (t->type == GGML_TYPE_F16) return fp16_to_fp32(peek_unaligned<uint16_t>((uint8_t*)t->data + i * 2));
    if (t->type == GGML_TYPE_Q8_0) {
        int block_idx = i / 32;
        int elem_idx = i % 32;
        uint8_t* block_ptr = (uint8_t*)t->data + block_idx * 34;
        uint16_t d_f16 = peek_unaligned<uint16_t>(block_ptr);
        int8_t q = *(int8_t*)(block_ptr + 2 + elem_idx);
        return fp16_to_fp32(d_f16) * q;
    }
    return 0.0f;
}

TensorInfo* get_tensor_or_fail(GGUFModel& model, const std::string& name) {
    TensorInfo* t = model.get_tensor(name);
    if (!t) { std::cerr << "FATAL: Missing tensor: " << name << std::endl; exit(1); }
    return t;
}

// CPU Vector Ops (NEON)
void vec_add(float* dst, const float* a, const float* b, int n) {
    int i = 0;
#if defined(__ARM_NEON)
    for (; i <= n - 4; i += 4) vst1q_f32(dst + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#endif
    for (; i < n; ++i) dst[i] = a[i] + b[i];
}

void vec_mul(float* dst, const float* a, const float* b, int n) {
    int i = 0;
#if defined(__ARM_NEON)
    for (; i <= n - 4; i += 4) vst1q_f32(dst + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#endif
    for (; i < n; ++i) dst[i] = a[i] * b[i];
}

void rms_norm(float* dst, const float* src, TensorInfo* weight_t, int n, float eps) {
    float sum = 0.0f;
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i <= n - 4; i += 4) {
        float32x4_t val = vld1q_f32(src + i);
        vsum = vmlaq_f32(vsum, val, val);
    }
    float32x2_t vpair = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    vpair = vpadd_f32(vpair, vpair);
    sum = vget_lane_f32(vpair, 0);
#endif
    for (; i < n; ++i) sum += src[i] * src[i];
    float scale = 1.0f / sqrt(sum / n + eps);
    for(i = 0; i < n; ++i) dst[i] = src[i] * scale * get_tensor_f32(weight_t, i);
}

void softmax(float* x, int n) {
    float max_val = x[0];
    for(int i=1; i<n; ++i) if(x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for(int i = 0; i < n; ++i) { x[i] = exp(x[i] - max_val); sum += x[i]; }
    float scale = 1.0f / sum;
    vec_mul(x, x, std::vector<float>(n, scale).data(), n); 
}

void silu(float* x, int n) {
    for(int i=0; i<n; ++i) {
        float sig = 1.0f / (1.0f + exp(-x[i]));
        x[i] = x[i] * sig;
    }
}

void rope(float* x, int n_dim, int n_head, int pos, float freq_base) {
    int head_dim = n_dim / n_head;
    for (int h = 0; h < n_head; ++h) {
        for (int i = 0; i < head_dim; i += 2) {
            float theta = pos * powf(freq_base, -((float)i / head_dim));
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);
            float* ptr = x + h * head_dim + i;
            float v0 = ptr[0];
            float v1 = ptr[1];
            ptr[0] = v0 * cos_theta - v1 * sin_theta;
            ptr[1] = v0 * sin_theta + v1 * cos_theta;
        }
    }
}

// CPU Q8_0 Dot Product (NEON)
float vec_dot_q8_0_f32_neon(const void* vb, const float* vx, int n) {
    float sumf = 0.0f;
    int nb = n / 32;
    const uint8_t* pb = (const uint8_t*)vb;
    const float* px = vx;

#if defined(__ARM_NEON)
    float32x4_t sumv = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; ++i) {
        uint16_t d_f16; memcpy(&d_f16, pb, 2);
        float d = fp16_to_fp32(d_f16);
        const int8_t* qs = (const int8_t*)(pb + 2);
        
        for (int j = 0; j < 32; j += 16) {
            int8x16_t w_int8 = vld1q_s8(qs + j);
            int16x8_t w_low = vmovl_s8(vget_low_s8(w_int8));
            int16x8_t w_high = vmovl_s8(vget_high_s8(w_int8));
            float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_low)));
            float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_low)));
            float32x4_t w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_high)));
            float32x4_t w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_high)));
            float32x4_t x0 = vld1q_f32(px + j);
            float32x4_t x1 = vld1q_f32(px + j + 4);
            float32x4_t x2 = vld1q_f32(px + j + 8);
            float32x4_t x3 = vld1q_f32(px + j + 12);
            sumv = vmlaq_f32(sumv, w0, x0);
            sumv = vmlaq_f32(sumv, w1, x1);
            sumv = vmlaq_f32(sumv, w2, x2);
            sumv = vmlaq_f32(sumv, w3, x3);
        }
        float32x2_t vpair = vadd_f32(vget_low_f32(sumv), vget_high_f32(sumv));
        vpair = vpadd_f32(vpair, vpair);
        sumf += vget_lane_f32(vpair, 0) * d;
        sumv = vdupq_n_f32(0.0f);
        pb += 34; px += 32;
    }
    return sumf;
#else
    for (int i = 0; i < nb; ++i) {
        uint16_t d_f16 = peek_unaligned<uint16_t>(pb);
        float d = fp16_to_fp32(d_f16);
        const int8_t* qs = (const int8_t*)(pb + 2);
        float block_sum = 0.0f;
        for (int j = 0; j < 32; ++j) block_sum += qs[j] * px[j];
        sumf += block_sum * d;
        pb += 34; px += 32;
    }
    return sumf;
#endif
}

// Tuned thresholds based on VideoCore IV characteristics
#if HAS_OPENCL
const int64_t GPU_MIN_OPS = 1024 * 576;  // ~590K ops minimum for GPU
#endif

// Matrix Multiplication (Hybrid CPU/GPU)
void mat_vec_mul(float* dst, const float* x, TensorInfo* w, int m, int k, ThreadPool& pool, GGUFModel& model) {
    #if HAS_OPENCL
    int64_t ops = (int64_t)m * k;

    // GPU path: Only for large operations with cached buffers
    if (g_use_gpu && g_cl_initialized &&
        w->type == GGML_TYPE_Q8_0 &&
        model.gpu_buffers.count(w->name) &&
        ops >= GPU_MIN_OPS) {
        
        g_perf.gpu_calls++;
        if (gpu_gemv(dst, x, model.gpu_buffers[w->name], m, k)) {
            return;  // GPU succeeded
        }
        // Fall through to CPU on failure
    }
    #endif

    g_perf.cpu_calls++;
    auto t0 = time_us();

    // CPU Path (NEON Parallel)
    pool.parallel_for(0, m, [&](int start, int end) {
        for (int row = start; row < end; ++row) {
            float sum = 0.0f;
            if (w->type == GGML_TYPE_Q8_0) {
                size_t row_bytes = (k / 32) * 34;
                const uint8_t* w_row = (const uint8_t*)w->data + row * row_bytes;
                sum = vec_dot_q8_0_f32_neon(w_row, x, k);
            } else if (w->type == GGML_TYPE_F32) {
                const float* w_row = (const float*)w->data + row * k;
                int i = 0;
                #if defined(__ARM_NEON)
                float32x4_t vsum = vdupq_n_f32(0.0f);
                for (; i <= k - 4; i += 4) {
                    vsum = vmlaq_f32(vsum, vld1q_f32(x + i), vld1q_f32(w_row + i));
                }
                float32x2_t vpair = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                vpair = vpadd_f32(vpair, vpair);
                sum = vget_lane_f32(vpair, 0);
                #endif
                for (; i < k; ++i) sum += x[i] * w_row[i];
            } else {
                for (int col = 0; col < k; ++col) sum += x[col] * get_tensor_f32(w, row * k + col);
            }
            dst[row] = sum;
        }
    });

    auto t1 = time_us();
    g_perf.cpu_compute_us += (t1 - t0);
}

// =================================================================================================
// 7. Inference Loop
// =================================================================================================

struct InferenceState {
    std::vector<float> x, xb, hb, q, k, v, logits;
    std::vector<float> k_cache, v_cache;
    int head_dim;
};

void inference(GGUFModel& model, const std::vector<int>& input_tokens, int n_predict, int n_threads) {
    ThreadPool pool(n_threads);
    ModelConfig& cfg = model.config;
    TensorNaming& naming = model.naming;

    int dim = cfg.n_embd;
    int hidden_dim = 0; 
    int head_dim = dim / cfg.n_head;
    
    char buf[256];
    sprintf(buf, naming.layer_prefix.c_str(), 0);
    std::string l0_prefix = buf;
    
    TensorInfo* gate = model.get_tensor(l0_prefix + naming.ffn_gate);
    if(!gate) gate = model.get_tensor(l0_prefix + naming.ffn_up); 
    if (gate) hidden_dim = gate->ne[1]; 
    else { std::cerr << "FATAL: Hidden dim detection failed" << std::endl; exit(1); }
    
    std::cout << "Inference init: dim=" << dim << ", hidden=" << hidden_dim << ", head_dim=" << head_dim 
              << ", threads=" << n_threads << ", gpu=" << (g_use_gpu ? "ON" : "OFF") << std::endl;

    InferenceState state;
    state.x.resize(dim); state.xb.resize(dim); state.hb.resize(hidden_dim);
    state.q.resize(dim); state.k.resize(dim); state.v.resize(dim);
    state.logits.resize(cfg.vocab_size); state.head_dim = head_dim;
    
    size_t kv_size = (size_t)cfg.n_layer * cfg.n_ctx * cfg.n_head_kv * head_dim;
    state.k_cache.resize(kv_size, 0.0f); state.v_cache.resize(kv_size, 0.0f);

    std::vector<int> tokens = input_tokens;
    int n_processed = 0;
    int pos = 0;
    
    while (n_processed < input_tokens.size() + n_predict) {
        int token = (n_processed < input_tokens.size()) ? input_tokens[n_processed] : tokens.back();
        if (token == model.token_eos && n_processed >= input_tokens.size()) break;
        if (n_processed >= input_tokens.size()) { std::cout << detokenize(model, token) << std::flush; }

        TensorInfo* t_embd = get_tensor_or_fail(model, naming.token_embd);
        for(int i=0; i<dim; ++i) state.x[i] = get_tensor_f32(t_embd, token * dim + i);

        for (int l = 0; l < cfg.n_layer; ++l) {
            sprintf(buf, naming.layer_prefix.c_str(), l);
            std::string layer_prefix = buf;
            
            rms_norm(state.xb.data(), state.x.data(), get_tensor_or_fail(model, layer_prefix + naming.attn_norm), dim, cfg.rms_norm_eps);
            
            mat_vec_mul(state.q.data(), state.xb.data(), get_tensor_or_fail(model, layer_prefix + naming.attn_q), cfg.n_head * head_dim, dim, pool, model);
            mat_vec_mul(state.k.data(), state.xb.data(), get_tensor_or_fail(model, layer_prefix + naming.attn_k), cfg.n_head_kv * head_dim, dim, pool, model);
            mat_vec_mul(state.v.data(), state.xb.data(), get_tensor_or_fail(model, layer_prefix + naming.attn_v), cfg.n_head_kv * head_dim, dim, pool, model);
            
            rope(state.q.data(), cfg.n_head * head_dim, cfg.n_head, pos, cfg.rope_freq_base);
            rope(state.k.data(), cfg.n_head_kv * head_dim, cfg.n_head_kv, pos, cfg.rope_freq_base);
            
            size_t cache_offset = l * cfg.n_ctx * cfg.n_head_kv * head_dim + pos * cfg.n_head_kv * head_dim;
            if (cache_offset + cfg.n_head_kv * head_dim <= state.k_cache.size()) {
                memcpy(state.k_cache.data() + cache_offset, state.k.data(), cfg.n_head_kv * head_dim * sizeof(float));
                memcpy(state.v_cache.data() + cache_offset, state.v.data(), cfg.n_head_kv * head_dim * sizeof(float));
            }
            
            std::fill(state.xb.begin(), state.xb.end(), 0.0f);
            int kv_mul = cfg.n_head / cfg.n_head_kv;
            
            for (int h = 0; h < cfg.n_head; ++h) {
                int kv_h = h / kv_mul; 
                std::vector<float> scores(pos + 1);
                for (int p = 0; p <= pos; ++p) {
                    float score = 0.0f;
                    size_t p_cache_offset = l * cfg.n_ctx * cfg.n_head_kv * head_dim + p * cfg.n_head_kv * head_dim;
                    float* k_ptr = state.k_cache.data() + p_cache_offset + kv_h * head_dim;
                    float* q_ptr = state.q.data() + h * head_dim;
                    for (int i = 0; i < head_dim; ++i) score += q_ptr[i] * k_ptr[i];
                    scores[p] = score / sqrt((float)head_dim);
                }
                softmax(scores.data(), pos + 1);
                float* out_ptr = state.xb.data() + h * head_dim;
                for (int p = 0; p <= pos; ++p) {
                    size_t p_cache_offset = l * cfg.n_ctx * cfg.n_head_kv * head_dim + p * cfg.n_head_kv * head_dim;
                    float* v_ptr = state.v_cache.data() + p_cache_offset + kv_h * head_dim;
                    float weight = scores[p];
                    for (int i = 0; i < head_dim; ++i) out_ptr[i] += weight * v_ptr[i];
                }
            }
            
            std::vector<float> attn_out(dim);
            mat_vec_mul(attn_out.data(), state.xb.data(), get_tensor_or_fail(model, layer_prefix + naming.attn_out), dim, dim, pool, model);
            vec_add(state.x.data(), state.x.data(), attn_out.data(), dim);

            rms_norm(state.xb.data(), state.x.data(), get_tensor_or_fail(model, layer_prefix + naming.ffn_norm), dim, cfg.rms_norm_eps);
            
            mat_vec_mul(state.hb.data(), state.xb.data(), get_tensor_or_fail(model, layer_prefix + naming.ffn_gate), hidden_dim, dim, pool, model);
            std::vector<float> h_up(hidden_dim);
            mat_vec_mul(h_up.data(), state.xb.data(), get_tensor_or_fail(model, layer_prefix + naming.ffn_up), hidden_dim, dim, pool, model);
            
            silu(state.hb.data(), hidden_dim);
            vec_mul(state.hb.data(), state.hb.data(), h_up.data(), hidden_dim);
            
            std::vector<float> ffn_out(dim);
            mat_vec_mul(ffn_out.data(), state.hb.data(), get_tensor_or_fail(model, layer_prefix + naming.ffn_down), dim, hidden_dim, pool, model);
            vec_add(state.x.data(), state.x.data(), ffn_out.data(), dim);
        }

        rms_norm(state.x.data(), state.x.data(), get_tensor_or_fail(model, naming.output_norm), dim, cfg.rms_norm_eps);
        mat_vec_mul(state.logits.data(), state.x.data(), get_tensor_or_fail(model, naming.output), cfg.vocab_size, dim, pool, model);
        
        int best_token = 0; float max_logit = state.logits[0];
        for(int i=1; i<cfg.vocab_size; ++i) { if (state.logits[i] > max_logit) { max_logit = state.logits[i]; best_token = i; } }
        
        tokens.push_back(best_token);
        n_processed++; pos++;
    }
    std::cout << "\n";
    g_perf.report();
}

// =================================================================================================
// 8. Main CLI
// =================================================================================================

#ifdef VC4LLM_TEST
int run_inference_cli(int argc, char** argv) {
#else
int main(int argc, char** argv) {
#endif
    std::string model_path, prompt = "Hello world";
    int n_predict = 20, n_threads = 4;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i+1 < argc) model_path = argv[++i];
        else if (arg == "-p" && i+1 < argc) prompt = argv[++i];
        else if (arg == "-n" && i+1 < argc) n_predict = std::stoi(argv[++i]);
        else if (arg == "-t" && i+1 < argc) n_threads = std::stoi(argv[++i]);
        else if (arg == "-v") verbose = true;
        else if (arg == "-g" || arg == "--gpu") {
            #if HAS_OPENCL
            g_use_gpu = true;
            #else
            std::cerr << "Warning: --gpu requested but OpenCL not available at compile time.\n";
            std::cerr << "Rebuild with: g++ ... -lOpenCL\n";
            #endif
        }
    }

    if (model_path.empty()) {
        std::cerr << "Usage: vc4llm -m <model> [-p <prompt>] [-n <tokens>] [-t <threads>] [-g|--gpu] [-v]\n";
        return 1;
    }

    #if HAS_OPENCL
    if (g_use_gpu) {
        if (!init_opencl()) {
            std::cerr << "Warning: OpenCL initialization failed. Falling back to CPU.\n";
            g_use_gpu = false;
        }
    }
    #endif

    GGUFModel model;
    if (!model.load(model_path, verbose)) return 1;

    std::vector<int> tokens = tokenize(model, prompt);
    std::cout << "Prompt: '" << prompt << "' (" << tokens.size() << " tokens)" << std::endl;
    std::cout << "Generating " << n_predict << " tokens with " << n_threads << " threads..." << std::endl;

    auto t0 = time_us();
    inference(model, tokens, n_predict, n_threads);
    auto t1 = time_us();
    
    float duration = (t1 - t0) / 1e6f;
    std::cout << "Done. Time: " << std::fixed << std::setprecision(2) << duration << "s (" 
              << (tokens.size() + n_predict) / duration << " tok/s)" << std::endl;

    return 0;
}
