#include <cassert>
#include <cmath>
#include <iostream>

// Mock OpenCL for CPU testing
#define MOCK_OPENCL

// Include main implementation
#include "vc4llm.cpp"

void test_fp16_conversion() {
    // Test normal values
    assert(std::abs(fp16_to_fp32(0x3C00) - 1.0f) < 1e-6);  // 1.0
    assert(std::abs(fp16_to_fp32(0x4000) - 2.0f) < 1e-6);  // 2.0
    assert(std::abs(fp16_to_fp32(0xBC00) - (-1.0f)) < 1e-6);  // -1.0

    // Test zero
    assert(fp16_to_fp32(0x0000) == 0.0f);

    std::cout << "✓ FP16 conversion tests passed" << std::endl;
}

void test_q8_0_dot_product() {
    // Create test Q8_0 block
    uint8_t block[34];
    uint16_t* scale = (uint16_t*)block;
    int8_t* quants = (int8_t*)(block + 2);

    *scale = 0x3C00;  // 1.0 in fp16
    for (int i = 0; i < 32; i++) {
        quants[i] = 1;  // All ones
    }

    // Input vector: all ones
    float x[32];
    for (int i = 0; i < 32; i++) x[i] = 1.0f;

    // Expected: 32 * 1 * 1.0 = 32.0
    float result = vec_dot_q8_0_f32_neon(block, x, 32);
    assert(std::abs(result - 32.0f) < 1e-4);

    std::cout << "✓ Q8_0 dot product tests passed" << std::endl;
}

void test_threadpool() {
    ThreadPool pool(4);

    std::vector<int> results(100, 0);
    std::atomic<int> sum(0);

    pool.parallel_for(0, 100, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            results[i] = i;
            sum.fetch_add(i);
        }
    });

    // Sum of 0..99 = 4950
    assert(sum == 4950);

    for (int i = 0; i < 100; i++) {
        assert(results[i] == i);
    }

    std::cout << "✓ ThreadPool tests passed" << std::endl;
}

int main() {
    test_fp16_conversion();
    test_q8_0_dot_product();
    test_threadpool();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
