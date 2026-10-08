#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int N = 16;
constexpr int BLOCK = 8;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE);                                  \
    }                                                              \
} while (0)

__global__ void triple_values(const int *input, int *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int value = input[i];
#ifdef DEMO_BAD_MATH
        int result = 2 * value; // deliberate bug: task is to triple
#else
        int result = 3 * value;
#endif
        output[i] = result;
    }
}

int main() {
    int input[N], output[N];
    for (int i = 0; i < N; ++i) input[i] = i + 1;

    int *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(input)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(output)));
    CUDA_CHECK(cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice));

    triple_values<<<N / BLOCK, BLOCK>>>(d_input, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        int expected = 3 * input[i];
        if (output[i] != expected) ++mismatches;
        std::printf("i=%2d  input=%2d  expected=%2d  actual=%2d  %s\n",
                    i, input[i], expected, output[i],
                    output[i] == expected ? "OK" : "WRONG");
    }
    std::printf("Verification: %s (%d mismatches)\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
