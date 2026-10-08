#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

constexpr int N = 1 << 20;
constexpr int BLOCK = 256;
constexpr int REPEATS = 50;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE);                                  \
    }                                                              \
} while (0)

__global__ void saxpy(const float *a, const float *b, float *c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = 2.0f * a[i] + b[i];
}

int main() {
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);
    float *a = static_cast<float *>(std::malloc(bytes));
    float *b = static_cast<float *>(std::malloc(bytes));
    float *c = static_cast<float *>(std::malloc(bytes));
    if (!a || !b || !c) {
        std::fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&da), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&db), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dc), bytes));
    CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

    const int grid = (N + BLOCK - 1) / BLOCK;
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        saxpy<<<grid, BLOCK>>>(da, db, dc);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(c[i] - 4.0f) > 1e-5f) ++mismatches;
    }
    std::printf("Vector length: %d; kernel launches: %d\n", N, REPEATS);
    std::printf("Expected c[i] = 2 * 1 + 2 = 4\n");
    std::printf("Verification: %s (%d mismatches)\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    std::free(a);
    std::free(b);
    std::free(c);
    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
