/*
 * VC4LLM: VideoCore IV LLM Inference Engine
 * Phase 1: CPU-Only Foundation (MVP) - Bugfix Release #5
 *
 * Target Hardware: Raspberry Pi 3B (Cortex-A53 + VideoCore IV)
 * Current State: GGUF Loader + Naive Scalar Inference + BPE Decoding
 *
 * Compile: g++ -O3 -std=c++17 -o vc4llm vc4llm.cpp -lpthread
 */

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
    // ...
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // ...
    GGML_TYPE_Q2_K = 11,
    GGML_TYPE_Q3_K = 12,
    GGML_TYPE_Q4_K = 15, // Common in Q4_K_M
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 16,
    GGML_TYPE_COUNT,
};

// Simplified Q8_0 block structure
// Block size = 32. 
// Structure: delta (f16), weights (int8 x 32)
// Total bytes = 2 + 32 = 34
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
    uint64_t ne[4]; // number of elements in each dimension
    ggml_type type;
    uint64_t offset;
    void* data = nullptr; // Pointer to mapped data or buffer
    
    // Helper to calculate total elements
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
            case GGML_TYPE_Q8_0: return (n / Q8_0_BLOCK_SIZE) * 34; 
            case GGML_TYPE_Q4_0: return (n / 32) * 18; // 2 bytes min/delta + 16 bytes quants
            // Fallback for size estimation (not accurate for K-quants, but prevents div/0)
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

// Tensor naming convention mapping
struct TensorNaming {
    std::string token_embd;
    std::string output_norm;
    std::string output;
    
    // Patterns for layer tensors
    // Use printf style formatting (e.g. "blk.%d.attn_q.weight")
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

struct Token {
    int id;
    std::string text;
    float score;
};

// =================================================================================================
// 2. Utils
// =================================================================================================

uint64_t time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// Basic half-precision to float conversion (scalar)
float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 0x1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;

    if (e == 0) {
        if (m == 0) return (s ? -0.0f : 0.0f);
        return (s ? -1.0f : 1.0f) * std::ldexp((float)m, -24);
    } else if (e == 31) {
        return (s ? -1.0f : 1.0f) * (m ? NAN : INFINITY);
    }

    return (s ? -1.0f : 1.0f) * std::ldexp((float)(m + 1024), e - 25);
}

// Safe unaligned read helper (replaces direct reinterpret_cast)
template<typename T>
T read_unaligned(uint8_t*& ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    ptr += sizeof(T);
    return val;
}

// Version that doesn't advance pointer
template<typename T>
T peek_unaligned(const uint8_t* ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    return val;
}

// =================================================================================================
// 3. GGUF Parser
// =================================================================================================

class GGUFModel {
public:
    ModelConfig config;
    TensorNaming naming;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, TensorInfo*> tensor_map;
    std::vector<char> model_data; 
    
    // Vocab
    std::vector<std::string> vocab_tokens;
    std::vector<float> vocab_scores;
    std::map<std::string, int> token_to_id;
    int token_bos = 1;
    int token_eos = 2;

    bool load(const std::string& path, bool verbose = false) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << path << std::endl;
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::cout << "Loading model: " << path << " (" << size / 1024 / 1024 << " MB)..." << std::endl;
        
        // Allocate memory
        try {
            model_data.resize(size);
        } catch (const std::exception& e) {
            std::cerr << "Error: Failed to allocate memory for model (" << e.what() << ")" << std::endl;
            return false;
        }

        if (!file.read(model_data.data(), size)) {
            std::cerr << "Error: Failed to read file data" << std::endl;
            return false;
        }

        uint8_t* ptr = reinterpret_cast<uint8_t*>(model_data.data());
        
        // Header (Safe Read)
        GGUFHeader header;
        header.magic = read_unaligned<uint32_t>(ptr);
        header.version = read_unaligned<uint32_t>(ptr);
        header.tensor_count = read_unaligned<uint64_t>(ptr);
        header.kv_count = read_unaligned<uint64_t>(ptr);

        if (header.magic != GGUF_MAGIC) {
            std::cerr << "Error: Invalid GGUF magic" << std::endl;
            return false;
        }
        if (header.version != GGUF_VERSION) {
            std::cerr << "Error: Unsupported GGUF version " << header.version << std::endl;
            return false;
        }

        std::cout << "GGUF v3, Tensors: " << header.tensor_count << ", KV: " << header.kv_count << std::endl;

        // KV Pairs
        for (uint64_t i = 0; i < header.kv_count; ++i) {
            std::string key = read_string(ptr);
            uint32_t type = read_unaligned<uint32_t>(ptr);

            if (verbose) {
                std::cout << "  KV: " << key << " (type=" << type << ")" << std::endl;
            }
            
            // --- Basic Config ---
            if (key == "llama.embedding_length" || key == "gpt2.embedding_length") config.n_embd = read_val<uint32_t>(ptr, type);
            else if (key == "llama.block_count" || key == "gpt2.block_count") config.n_layer = read_val<uint32_t>(ptr, type);
            else if (key == "llama.attention.head_count" || key == "gpt2.attention.head_count") config.n_head = read_val<uint32_t>(ptr, type);
            else if (key == "llama.attention.head_count_kv") config.n_head_kv = read_val<uint32_t>(ptr, type);
            else if (key == "llama.context_length" || key == "gpt2.context_length") config.n_ctx = read_val<uint32_t>(ptr, type);
            else if (key == "llama.rope.freq_base") config.rope_freq_base = read_val<float>(ptr, type);
            else if (key == "tokenizer.ggml.bos_token_id") token_bos = read_val<uint32_t>(ptr, type);
            else if (key == "tokenizer.ggml.eos_token_id") token_eos = read_val<uint32_t>(ptr, type);
            
            // --- Arrays (FIXED & Safe) ---
            else if (key == "tokenizer.ggml.tokens") {
                uint32_t item_type = read_unaligned<uint32_t>(ptr); 
                uint64_t count = read_unaligned<uint64_t>(ptr);
                
                // Sanity check
                if (count > 200000) {
                    std::cerr << "Error: Vocab size too large (" << count << ")" << std::endl;
                    return false;
                }

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
                for(uint64_t j=0; j<count; ++j) {
                    vocab_scores[j] = read_unaligned<float>(ptr);
                }
            }
            else {
                skip_val(ptr, type);
            }
        }

        if (config.n_head_kv == 0) config.n_head_kv = config.n_head; // Handle MHA

        // Tensor Infos
        for (uint64_t i = 0; i < header.tensor_count; ++i) {
            TensorInfo info;
            info.name = read_string(ptr);
            info.n_dims = read_unaligned<uint32_t>(ptr);
            for(uint32_t d=0; d<info.n_dims; ++d) {
                info.ne[d] = read_unaligned<uint64_t>(ptr);
            }
            info.type = (ggml_type)read_unaligned<uint32_t>(ptr);
            info.offset = read_unaligned<uint64_t>(ptr);
            
            tensors.push_back(info);
        }

        // Align to 64 bytes for tensor data start
        uint64_t offset_base = (reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(model_data.data()));
        offset_base = (offset_base + 63) & ~63; 

        for (auto& t : tensors) {
            t.data = model_data.data() + offset_base + t.offset;
            tensor_map[t.name] = &t;
        }

        discover_names();

        std::cout << "Model loaded. " << config.n_layer << " layers, " << config.n_embd << " dim, " << config.n_head << " heads." << std::endl;
        return true;
    }

    TensorInfo* get_tensor(const std::string& name) {
        if (tensor_map.find(name) != tensor_map.end()) return tensor_map[name];
        return nullptr;
    }

private:
    void discover_names() {
        // Detect naming convention (Standard vs HuggingFace/SmolLM)
        
        bool has_blk = (tensor_map.find("blk.0.attn_q.weight") != tensor_map.end());
        bool has_model = (tensor_map.find("model.layers.0.self_attn.q_proj.weight") != tensor_map.end());
        
        if (has_blk) {
            // Standard GGUF (llama.cpp style)
            naming.token_embd = "token_embd.weight";
            naming.output_norm = "output_norm.weight";
            naming.layer_prefix = "blk.%d.";
            naming.attn_norm = "attn_norm.weight";
            naming.attn_q = "attn_q.weight";
            naming.attn_k = "attn_k.weight";
            naming.attn_v = "attn_v.weight";
            naming.attn_out = "attn_output.weight";
            naming.ffn_norm = "ffn_norm.weight";
            naming.ffn_gate = "ffn_gate.weight";
            naming.ffn_up = "ffn_up.weight";
            naming.ffn_down = "ffn_down.weight";
            
            // Check for output weight (or tied embedding)
            if (tensor_map.find("output.weight") != tensor_map.end()) {
                naming.output = "output.weight";
            } else if (tensor_map.find("token_embd.weight") != tensor_map.end()) {
                naming.output = "token_embd.weight"; // Tied
                std::cout << "Naming: Tied embeddings detected (output = token_embd.weight)" << std::endl;
            } else {
                naming.output = "output.weight"; // Fail likely
            }

            std::cout << "Naming convention: Standard (blk.X)" << std::endl;
        } else if (has_model) {
            // HuggingFace / SmolLM style
            naming.token_embd = "model.embed_tokens.weight";
            naming.output_norm = "model.norm.weight";
            naming.layer_prefix = "model.layers.%d.";
            naming.attn_norm = "input_layernorm.weight";
            naming.attn_q = "self_attn.q_proj.weight";
            naming.attn_k = "self_attn.k_proj.weight";
            naming.attn_v = "self_attn.v_proj.weight";
            naming.attn_out = "self_attn.o_proj.weight";
            naming.ffn_norm = "post_attention_layernorm.weight";
            naming.ffn_gate = "mlp.gate_proj.weight";
            naming.ffn_up = "mlp.up_proj.weight";
            naming.ffn_down = "mlp.down_proj.weight";
            
            // Check for output weight (or tied embedding)
            if (tensor_map.find("lm_head.weight") != tensor_map.end()) {
                naming.output = "lm_head.weight";
            } else if (tensor_map.find("output.weight") != tensor_map.end()) {
                naming.output = "output.weight";
            } else if (tensor_map.find("model.embed_tokens.weight") != tensor_map.end()) {
                naming.output = "model.embed_tokens.weight"; // Tied
                std::cout << "Naming: Tied embeddings detected (output = model.embed_tokens.weight)" << std::endl;
            } else {
                 naming.output = "lm_head.weight"; // Fail likely
            }
            
             std::cout << "Naming convention: HuggingFace (model.layers.X)" << std::endl;
        } else {
             // Fallback or Unknown - defaults to Standard, logic might fail
             std::cerr << "WARNING: Unknown tensor naming convention. Inference will likely fail." << std::endl;
             naming.token_embd = "token_embd.weight";
             naming.output = "output.weight";
             naming.layer_prefix = "blk.%d.";
             // ...
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
            default: break; 
        }
        return val;
    }

    void skip_val(uint8_t*& ptr, uint32_t type) {
        switch(type) {
            case 0: case 1: case 7: ptr += 1; break;
            case 2: case 3: ptr += 2; break;
            case 4: case 5: case 6: ptr += 4; break;
            case 8: { uint64_t len = read_unaligned<uint64_t>(ptr); ptr += len; break; }
            case 9: { // Array
                uint32_t itype = read_unaligned<uint32_t>(ptr);
                uint64_t len = read_unaligned<uint64_t>(ptr);
                for(uint64_t i=0; i<len; ++i) skip_val(ptr, itype);
                break;
            }
            case 10: case 11: ptr += 8; break; 
            default: break;
        }
    }
};

// =================================================================================================
// 4. Tokenizer (Naive Greedy)
// =================================================================================================

std::vector<int> tokenize(GGUFModel& model, const std::string& text) {
    std::vector<int> tokens;
    std::string input = text;
    
    // Naive preprocessing: replace spaces
    if (model.token_bos != -1) tokens.push_back(model.token_bos);

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
        
        if (best_id == -1) {
            std::string c = input.substr(i, 1);
            if(model.token_to_id.count(c)) {
                best_id = model.token_to_id[c];
                best_len = 1;
            } else {
                 i++; continue;
            }
        }

        tokens.push_back(best_id);
        i += best_len;
    }
    return tokens;
}

// BPE Decoder Helpers
std::unordered_map<char32_t, uint8_t> build_byte_decoder() {
    std::unordered_map<char32_t, uint8_t> decoder;
    
    // Printable ASCII (33-126) maps to itself
    for (int b = 33; b <= 126; b++) {
        decoder[b] = b;
    }
    // Extended Latin (161-172, 174-255) maps to itself  
    for (int b = 161; b <= 172; b++) {
        decoder[b] = b;
    }
    for (int b = 174; b <= 255; b++) {
        decoder[b] = b;
    }
    
    // Non-printable bytes map to U+0100 + offset
    int n = 0;
    for (int b = 0; b < 256; b++) {
        // Check if already mapped above
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            continue;
        }
        // Map U+0100+n back to byte b
        decoder[256 + n] = b;
        n++;
    }
    
    return decoder;
}

// Decode a single UTF-8 character and return its code point
char32_t decode_utf8_char(const char*& ptr) {
    uint8_t c = *ptr++;
    
    if ((c & 0x80) == 0) {
        // ASCII (0xxxxxxx)
        return c;
    } else if ((c & 0xE0) == 0xC0) {
        // 2-byte sequence (110xxxxx 10xxxxxx)
        char32_t cp = (c & 0x1F) << 6;
        cp |= (*ptr++ & 0x3F);
        return cp;
    } else if ((c & 0xF0) == 0xE0) {
        // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
        char32_t cp = (c & 0x0F) << 12;
        cp |= (*ptr++ & 0x3F) << 6;
        cp |= (*ptr++ & 0x3F);
        return cp;
    } else if ((c & 0xF8) == 0xF0) {
        // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
        char32_t cp = (c & 0x07) << 18;
        cp |= (*ptr++ & 0x3F) << 12;
        cp |= (*ptr++ & 0x3F) << 6;
        cp |= (*ptr++ & 0x3F);
        return cp;
    }
    
    return 0xFFFD; // Replacement character for invalid UTF-8
}

static std::unordered_map<char32_t, uint8_t> g_byte_decoder;
static bool g_byte_decoder_initialized = false;

std::string detokenize(GGUFModel& model, int id) {
    if (!g_byte_decoder_initialized) {
        g_byte_decoder = build_byte_decoder();
        g_byte_decoder_initialized = true;
    }
    
    if (id < 0 || id >= model.vocab_tokens.size()) return "";
    
    const std::string& token = model.vocab_tokens[id];
    
    // Skip special tokens
    if (token == "<s>" || token == "</s>" || 
        token == "<unk>" || token == "<pad>" ||
        token == "<|endoftext|>") {
        return "";
    }

    std::string result;
    
    const char* ptr = token.c_str();
    const char* end = ptr + token.length();
    
    while (ptr < end) {
        char32_t cp = decode_utf8_char(ptr);
        
        auto it = g_byte_decoder.find(cp);
        if (it != g_byte_decoder.end()) {
            result += (char)it->second;
        } else {
            // Unknown code point - output as UTF-8
            // (This handles actual Unicode characters in the vocab)
            if (cp < 0x80) {
                result += (char)cp;
            } else if (cp < 0x800) {
                result += (char)(0xC0 | (cp >> 6));
                result += (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                result += (char)(0xE0 | (cp >> 12));
                result += (char)(0x80 | ((cp >> 6) & 0x3F));
                result += (char)(0x80 | (cp & 0x3F));
            }
        }
    }
    
    return result;
}

// =================================================================================================
// 5. Compute Primitives
// =================================================================================================

static std::set<int> unsupported_types_warned;

// Get float value from tensor at index i (auto-dequantize)
float get_tensor_f32(TensorInfo* t, int i) {
    if (t->type == GGML_TYPE_F32) {
        return ((float*)t->data)[i];
    } 
    else if (t->type == GGML_TYPE_F16) {
        uint16_t h = peek_unaligned<uint16_t>((uint8_t*)t->data + i * 2);
        return fp16_to_fp32(h);
    }
    else if (t->type == GGML_TYPE_Q8_0) {
        int block_idx = i / 32;
        int elem_idx = i % 32;
        uint8_t* block_ptr = (uint8_t*)t->data + block_idx * 34;
        
        uint16_t d_f16 = peek_unaligned<uint16_t>(block_ptr);
        float d = fp16_to_fp32(d_f16);
        int8_t q = *(int8_t*)(block_ptr + 2 + elem_idx);
        
        return d * q;
    }
    else {
        if (unsupported_types_warned.find(t->type) == unsupported_types_warned.end()) {
            std::cerr << "WARNING: Unsupported quantization type " << t->type 
                      << " for tensor " << t->name << ". Output will be garbage." << std::endl;
            unsupported_types_warned.insert(t->type);
        }
        return 0.0f; 
    }
}

// Helper to get tensor or exit with error (safe access)
TensorInfo* get_tensor_or_fail(GGUFModel& model, const std::string& name) {
    TensorInfo* t = model.get_tensor(name);
    if (!t) {
        std::cerr << "FATAL ERROR: Missing tensor: " << name << std::endl;
        std::cerr << "This typically means the model structure is not compatible or naming convention failed." << std::endl;
        exit(1);
    }
    return t;
}

void vec_add(float* dst, const float* a, const float* b, int n) {
    for(int i=0; i<n; ++i) dst[i] = a[i] + b[i];
}

void vec_mul(float* dst, const float* a, const float* b, int n) {
    for(int i=0; i<n; ++i) dst[i] = a[i] * b[i];
}

// Generic RMS Norm that handles different weight types (F32, F16)
void rms_norm(float* dst, const float* src, TensorInfo* weight_t, int n, float eps) {
    float sum = 0.0f;
    for(int i=0; i<n; ++i) sum += src[i] * src[i];
    float scale = 1.0f / sqrt(sum / n + eps);
    
    for(int i=0; i<n; ++i) {
        float w = get_tensor_f32(weight_t, i);
        dst[i] = src[i] * scale * w;
    }
}

void softmax(float* x, int n) {
    float max_val = x[0];
    for(int i=1; i<n; ++i) if(x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for(int i=0; i<n; ++i) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    for(int i=0; i<n; ++i) x[i] /= sum;
}

void silu(float* x, int n) {
    for(int i=0; i<n; ++i) {
        float sig = 1.0f / (1.0f + exp(-x[i]));
        x[i] = x[i] * sig;
    }
}

// Matrix Multiplication: y = x @ A^T
void mat_vec_mul(float* dst, const float* x, TensorInfo* w, int m, int k) {
    for (int row = 0; row < m; ++row) {
        float sum = 0.0f;
        
        if (w->type == GGML_TYPE_Q8_0) {
            int num_blocks = k / 32;
            uint8_t* w_data = (uint8_t*)w->data + row * num_blocks * 34;
            
            for (int b = 0; b < num_blocks; ++b) {
                // Safe read of float16 scale
                uint16_t d_f16 = peek_unaligned<uint16_t>(w_data);
                float d = fp16_to_fp32(d_f16);
                int8_t* qs = (int8_t*)(w_data + 2);
                
                for (int j = 0; j < 32; ++j) {
                    sum += x[b * 32 + j] * (d * qs[j]);
                }
                w_data += 34;
            }
        } 
        else if (w->type == GGML_TYPE_F32) {
             float* w_data = (float*)w->data + row * k;
             for (int col = 0; col < k; ++col) {
                 sum += x[col] * w_data[col];
             }
        }
        else {
            // Slow fallback for other types via get_tensor_f32
            for (int col = 0; col < k; ++col) {
                sum += x[col] * get_tensor_f32(w, row * k + col);
            }
        }
        dst[row] = sum;
    }
}

void rope(float* x, int n_dim, int n_head, int pos, float freq_base) {
    int head_dim = n_dim / n_head;
    for (int h = 0; h < n_head; ++h) {
        for (int i = 0; i < head_dim; i += 2) {
            float theta = pos * pow(freq_base, -((float)i / head_dim));
            float cos_theta = cos(theta);
            float sin_theta = sin(theta);
            
            float* ptr = x + h * head_dim + i;
            float v0 = ptr[0];
            float v1 = ptr[1];
            
            ptr[0] = v0 * cos_theta - v1 * sin_theta;
            ptr[1] = v0 * sin_theta + v1 * cos_theta;
        }
    }
}

// =================================================================================================
// 6. Inference Loop
// =================================================================================================

struct InferenceState {
    std::vector<float> x, xb, hb, q, k, v, logits;
    std::vector<float> k_cache, v_cache;
    int head_dim;
};

void inference(GGUFModel& model, const std::vector<int>& input_tokens, int n_predict) {
    ModelConfig& cfg = model.config;
    TensorNaming& naming = model.naming;

    int dim = cfg.n_embd;
    int hidden_dim = 0; 
    int head_dim = dim / cfg.n_head;
    
    // Dynamically resolve names
    char buf[256];
    sprintf(buf, naming.layer_prefix.c_str(), 0);
    std::string l0_prefix = buf;
    
    TensorInfo* gate = model.get_tensor(l0_prefix + naming.ffn_gate);
    if(!gate) gate = model.get_tensor(l0_prefix + naming.ffn_up); 
    
    if (gate) {
        hidden_dim = gate->ne[1]; // M dimension is typically hidden dim for linear layer
    } else {
        std::cerr << "FATAL: Could not determine hidden dimension from FFN tensors." << std::endl;
        exit(1);
    }
    
    std::cout << "Inference init: dim=" << dim << ", hidden=" << hidden_dim << ", head_dim=" << head_dim << std::endl;

    InferenceState state;
    state.x.resize(dim);
    state.xb.resize(dim);
    state.hb.resize(hidden_dim);
    state.q.resize(dim);
    state.k.resize(dim); 
    state.v.resize(dim);
    state.logits.resize(cfg.vocab_size);
    state.head_dim = head_dim;
    
    size_t kv_size = (size_t)cfg.n_layer * cfg.n_ctx * cfg.n_head_kv * head_dim;
    try {
        state.k_cache.resize(kv_size, 0.0f);
        state.v_cache.resize(kv_size, 0.0f);
    } catch(const std::exception& e) {
        std::cerr << "Error allocating KV cache (OOM?)" << std::endl;
        return;
    }

    std::vector<int> tokens = input_tokens;
    int n_processed = 0;
    int pos = 0;
    
    while (n_processed < input_tokens.size() + n_predict) {
        int token = (n_processed < input_tokens.size()) ? input_tokens[n_processed] : tokens.back();
        if (token == model.token_eos && n_processed >= input_tokens.size()) break;
        
        if (n_processed >= input_tokens.size()) {
            std::cout << detokenize(model, token) << std::flush;
        }

        // Embedding
        TensorInfo* t_embd = get_tensor_or_fail(model, naming.token_embd);
        for(int i=0; i<dim; ++i) {
            state.x[i] = get_tensor_f32(t_embd, token * dim + i);
        }

        // Layers
        for (int l = 0; l < cfg.n_layer; ++l) {
            sprintf(buf, naming.layer_prefix.c_str(), l);
            std::string layer_prefix = buf;
            
            // Attn Norm
            rms_norm(state.xb.data(), state.x.data(), 
                     get_tensor_or_fail(model, layer_prefix + naming.attn_norm), 
                     dim, cfg.rms_norm_eps);
            
            // QKV
            TensorInfo* wq = get_tensor_or_fail(model, layer_prefix + naming.attn_q);
            TensorInfo* wk = get_tensor_or_fail(model, layer_prefix + naming.attn_k);
            TensorInfo* wv = get_tensor_or_fail(model, layer_prefix + naming.attn_v);
            
            mat_vec_mul(state.q.data(), state.xb.data(), wq, cfg.n_head * head_dim, dim);
            mat_vec_mul(state.k.data(), state.xb.data(), wk, cfg.n_head_kv * head_dim, dim);
            mat_vec_mul(state.v.data(), state.xb.data(), wv, cfg.n_head_kv * head_dim, dim);
            
            // RoPE
            rope(state.q.data(), cfg.n_head * head_dim, cfg.n_head, pos, cfg.rope_freq_base);
            rope(state.k.data(), cfg.n_head_kv * head_dim, cfg.n_head_kv, pos, cfg.rope_freq_base);
            
            // KV Cache Update
            size_t cache_offset = l * cfg.n_ctx * cfg.n_head_kv * head_dim + pos * cfg.n_head_kv * head_dim;
            if (cache_offset + cfg.n_head_kv * head_dim <= state.k_cache.size()) {
                memcpy(state.k_cache.data() + cache_offset, state.k.data(), cfg.n_head_kv * head_dim * sizeof(float));
                memcpy(state.v_cache.data() + cache_offset, state.v.data(), cfg.n_head_kv * head_dim * sizeof(float));
            }
            
            // Attention (MHA/GQA)
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
            
            // Output
            TensorInfo* wo = get_tensor_or_fail(model, layer_prefix + naming.attn_out);
            std::vector<float> attn_out(dim);
            mat_vec_mul(attn_out.data(), state.xb.data(), wo, dim, dim);
            vec_add(state.x.data(), state.x.data(), attn_out.data(), dim);
            
            // FFN
            rms_norm(state.xb.data(), state.x.data(), 
                     get_tensor_or_fail(model, layer_prefix + naming.ffn_norm), 
                     dim, cfg.rms_norm_eps);
                     
            TensorInfo* w_gate = get_tensor_or_fail(model, layer_prefix + naming.ffn_gate);
            TensorInfo* w_up = get_tensor_or_fail(model, layer_prefix + naming.ffn_up);
            TensorInfo* w_down = get_tensor_or_fail(model, layer_prefix + naming.ffn_down);
            
            mat_vec_mul(state.hb.data(), state.xb.data(), w_gate, hidden_dim, dim); // reuse hb for gate
            std::vector<float> h_up(hidden_dim);
            mat_vec_mul(h_up.data(), state.xb.data(), w_up, hidden_dim, dim);
            
            silu(state.hb.data(), hidden_dim);
            vec_mul(state.hb.data(), state.hb.data(), h_up.data(), hidden_dim);
            
            std::vector<float> ffn_out(dim);
            mat_vec_mul(ffn_out.data(), state.hb.data(), w_down, dim, hidden_dim);
            vec_add(state.x.data(), state.x.data(), ffn_out.data(), dim);
        }

        // Output
        rms_norm(state.x.data(), state.x.data(), 
                 get_tensor_or_fail(model, naming.output_norm), 
                 dim, cfg.rms_norm_eps);
                 
        TensorInfo* w_out = get_tensor_or_fail(model, naming.output);
        
        // Debug info (once)
        static bool output_debugged = false;
        if (!output_debugged) {
            std::cout << "Output tensor: " << w_out->name 
                      << " shape=[" << w_out->ne[0] << "," << w_out->ne[1] << "]"
                      << " type=" << (int)w_out->type << std::endl;
            output_debugged = true;
        }

        mat_vec_mul(state.logits.data(), state.x.data(), w_out, cfg.vocab_size, dim);
        
        // Argmax
        int best_token = 0;
        float max_logit = state.logits[0];
        for(int i=1; i<cfg.vocab_size; ++i) {
            if (state.logits[i] > max_logit) {
                max_logit = state.logits[i];
                best_token = i;
            }
        }
        
        tokens.push_back(best_token);
        n_processed++;
        pos++;
    }
    std::cout << "\n";
}

// =================================================================================================
// 7. Main CLI
// =================================================================================================

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt = "Hello world";
    int n_predict = 20;
    bool verbose = false;
    bool list_tensors = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i+1 < argc) model_path = argv[++i];
        else if (arg == "-p" && i+1 < argc) prompt = argv[++i];
        else if (arg == "-n" && i+1 < argc) n_predict = std::stoi(argv[++i]);
        else if (arg == "--verbose" || arg == "-v") verbose = true;
        else if (arg == "--list-tensors") list_tensors = true;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: vc4llm -m <model.gguf> [-p <prompt>] [-n <tokens>] [-v] [--list-tensors]\n";
            return 0;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: Model path required (-m)\n";
        return 1;
    }

    GGUFModel model;
    if (!model.load(model_path, verbose)) return 1;

    if (list_tensors) {
        std::cout << "Tensors in model:" << std::endl;
        for (const auto& t : model.tensors) {
            std::cout << "  " << t.name 
                      << " [";
            for(int d=0; d<t.n_dims; d++) std::cout << (d>0?"x":"") << t.ne[d];
            std::cout << "] type=" << t.type << std::endl;
        }
        return 0;
    }

    std::vector<int> tokens = tokenize(model, prompt);
    
    std::cout << "Prompt: '" << prompt << "' (" << tokens.size() << " tokens)" << std::endl;
    std::cout << "Generating " << n_predict << " tokens..." << std::endl;

    auto t0 = time_us();
    inference(model, tokens, n_predict);
    auto t1 = time_us();
    
    float duration = (t1 - t0) / 1e6f;
    std::cout << "Done. Time: " << std::fixed << std::setprecision(2) << duration << "s (" 
              << (tokens.size() + n_predict) / duration << " tok/s)" << std::endl;

    return 0;
}
